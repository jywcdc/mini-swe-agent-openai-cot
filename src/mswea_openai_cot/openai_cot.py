from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from typing import Any

from openai import OpenAI
from openai import (
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
)
from pydantic import Field

from minisweagent.exceptions import FormatError
from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from minisweagent.models.utils.actions_toolcall_response import (
    BASH_TOOL_RESPONSE_API,
    format_toolcall_observation_messages,
    parse_toolcall_actions_response,
)
from minisweagent.models.utils.retry import retry

logger = logging.getLogger("openai_cot_model")

_ADAPTER_MODEL_KWARG_KEYS = {
    "api_model_name",
    "api_key_env",
    "base_url",
    "organization",
    "project",
    "client_kwargs",
    "drop_params",
    "log_raw_requests",
    "input_cost_per_token",
    "output_cost_per_token",
    "cached_input_cost_per_token",
    "cache_read_input_token_cost",
}


class OpenAIResponsesCoTModelConfig(LitellmModelConfig):
    """Configuration for a direct OpenAI Responses API mini-swe-agent model.

    ``model_name`` remains the mini-swe-agent/Pier model identity. Use
    ``api_model_name`` when the actual OpenAI Responses model slug differs.
    """

    api_model_name: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str | None = None
    organization: str | None = None
    project: str | None = None
    client_kwargs: dict[str, Any] = Field(default_factory=dict)
    log_raw_requests: bool = False
    input_cost_per_token: float | None = None
    output_cost_per_token: float | None = None
    cached_input_cost_per_token: float | None = None
    cache_read_input_token_cost: float | None = None


class OpenAIResponsesCoTModel(LitellmModel):
    """Stateful OpenAI Responses API adapter for mini-swe-agent.

    mini-swe-agent gives every model call the full local trajectory. This adapter
    preserves that full Responses API input while also setting
    ``previous_response_id`` after the first OpenAI response.
    """

    abort_exceptions = [
        *LitellmModel.abort_exceptions,
        AuthenticationError,
        BadRequestError,
        PermissionDeniedError,
        NotFoundError,
        KeyboardInterrupt,
    ]

    def __init__(
        self,
        *,
        config_class: Callable = OpenAIResponsesCoTModelConfig,
        client: Any | None = None,
        **kwargs: Any,
    ):
        super().__init__(config_class=config_class, **kwargs)
        self._client = client or self._build_client()
        self._previous_response_id: str | None = None

    def _build_client(self) -> OpenAI:
        client_kwargs = dict(self._setting("client_kwargs") or {})
        if base_url := self._setting("base_url"):
            client_kwargs["base_url"] = base_url
        if organization := self._setting("organization"):
            client_kwargs["organization"] = organization
        if project := self._setting("project"):
            client_kwargs["project"] = project
        api_key_env = self._setting("api_key_env")
        if api_key_env and os.getenv(api_key_env):
            client_kwargs["api_key"] = os.environ[api_key_env]
        return OpenAI(**client_kwargs)

    def _setting(self, name: str) -> Any:
        if name in self.config.model_kwargs:
            return self.config.model_kwargs[name]
        return getattr(self.config, name)

    @property
    def api_model_name(self) -> str:
        if api_model_name := self._setting("api_model_name"):
            return api_model_name
        if self.config.model_name.startswith("openai/"):
            return self.config.model_name.removeprefix("openai/")
        return self.config.model_name

    @staticmethod
    def _dump_response(response: Any) -> dict[str, Any]:
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json", exclude_none=True)
        if isinstance(response, dict):
            return {k: v for k, v in response.items() if v is not None}
        return dict(response)

    @staticmethod
    def _jsonable(value: Any) -> Any:
        return json.loads(json.dumps(value, default=str))

    @staticmethod
    def _strip_extra(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                k: OpenAIResponsesCoTModel._strip_extra(v)
                for k, v in value.items()
                if k != "extra" and v is not None
            }
        if isinstance(value, list):
            return [OpenAIResponsesCoTModel._strip_extra(v) for v in value]
        return value

    def _prepare_messages_for_api(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove mini-swe-agent metadata and flatten response objects if needed."""
        prepared: list[dict[str, Any]] = []
        for message in messages:
            if message.get("role") == "exit":
                continue
            if message.get("object") == "response":
                for item in message.get("output") or []:
                    if isinstance(item, dict):
                        prepared.append(self._strip_extra(item))
                continue
            prepared.append(self._strip_extra(message))
        return prepared

    def _input_for_request(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self._prepare_messages_for_api(messages)

    def _reasoning_for_request(self, raw_kwargs: Mapping[str, Any]) -> Any:
        reasoning = raw_kwargs.get("reasoning") or {}
        if not isinstance(reasoning, dict):
            raise TypeError("model_kwargs.reasoning must be a mapping")
        result = dict(reasoning)
        result["context"] = "all_turns"
        return result

    def _build_request_kwargs(self, input_messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        raw_kwargs = {**self.config.model_kwargs, **kwargs}
        reasoning = self._reasoning_for_request(raw_kwargs)

        request_kwargs = {
            key: value
            for key, value in raw_kwargs.items()
            if key
            not in {
                "model",
                "input",
                "tools",
                "include",
                "previous_response_id",
                "reasoning",
                "store",
                *_ADAPTER_MODEL_KWARG_KEYS,
            }
        }
        extra_body = dict(request_kwargs.pop("extra_body", {}) or {})

        extra_body["reasoning"] = reasoning

        request_kwargs.update(
            {
                "model": self.api_model_name,
                "input": input_messages,
                "tools": [BASH_TOOL_RESPONSE_API],
                "include": ["reasoning.encrypted_content"],
                "store": False,
            }
        )

        if self._previous_response_id:
            request_kwargs["previous_response_id"] = self._previous_response_id
        if extra_body:
            request_kwargs["extra_body"] = extra_body

        return request_kwargs

    def _raw_request_for_log(self, request_kwargs: Mapping[str, Any]) -> dict[str, Any]:
        return self._jsonable(dict(request_kwargs))

    def _log_raw_request(self, request_log: Mapping[str, Any]) -> None:
        if self._setting("log_raw_requests"):
            logger.warning("OpenAI Responses API raw request: %s", json.dumps(request_log, sort_keys=True))

    def _query(self, request_kwargs: Mapping[str, Any], request_log: Mapping[str, Any]) -> Any:
        self._log_raw_request(request_log)
        try:
            return self._client.responses.create(**request_kwargs)
        except AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e

    def query(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        input_messages = self._input_for_request(messages)
        for attempt in retry(logger=logger, abort_exceptions=self.abort_exceptions):
            with attempt:
                request_previous_id = self._previous_response_id
                request_kwargs = self._build_request_kwargs(input_messages, **kwargs)
                request_log = self._raw_request_for_log(request_kwargs)
                response = self._query(request_kwargs, request_log)
                response_dict = self._dump_response(response)
                response_id = response_dict.get("id")
                if not response_id:
                    raise RuntimeError("OpenAI Responses API response did not include an id")

                cost_output = self._calculate_cost(response)
                GLOBAL_MODEL_STATS.add(cost_output["cost"])
                try:
                    actions = self._parse_actions(response)
                except FormatError as e:
                    self._remember_response(response_id)
                    try:
                        extra = e.messages[0].setdefault("extra", {})
                        extra["response"] = response_dict
                        extra["openai_cot"] = {
                            "request": request_log,
                            "request_previous_response_id": request_previous_id,
                            "response_id": response_id,
                        }
                    except Exception:
                        e.messages[0]["extra"] = {
                            "response": repr(response),
                            "openai_cot": {
                                "request": request_log,
                                "request_previous_response_id": request_previous_id,
                                "response_id": response_id,
                            },
                        }
                    raise

                self._remember_response(response_id)
                message = response_dict
                response_extra = dict(response_dict)
                message["extra"] = {
                    "actions": actions,
                    "response": response_extra,
                    **cost_output,
                    "timestamp": time.time(),
                    "openai_cot": {
                        "request": request_log,
                        "request_previous_response_id": request_previous_id,
                        "response_id": response_id,
                    },
                }
                return message

        raise RuntimeError("unreachable retry loop exit")

    def _remember_response(self, response_id: str) -> None:
        self._previous_response_id = response_id

    def _parse_actions(self, response: Any) -> list[dict[str, Any]]:
        output = getattr(response, "output", None)
        if output is None and isinstance(response, dict):
            output = response.get("output")
        return parse_toolcall_actions_response(
            output or [],
            format_error_template=self.config.format_error_template,
            template_kwargs={"finish_reason": finish_reason_from_responses_api(response)},
        )

    def _calculate_cost(self, response: Any) -> dict[str, float]:
        response_dict = self._dump_response(response)
        usage = response_dict.get("usage") or {}
        cost = usage.get("cost")
        if isinstance(cost, int | float) and cost > 0:
            return {"cost": float(cost)}

        input_rate = self._setting("input_cost_per_token")
        output_rate = self._setting("output_cost_per_token")
        cached_rate = self._setting("cached_input_cost_per_token")
        if cached_rate is None:
            cached_rate = self._setting("cache_read_input_token_cost")

        if input_rate is None or output_rate is None:
            raise RuntimeError(
                "No cost was returned by OpenAI and input/output token rates were not configured. "
                "Provide input_cost_per_token/output_cost_per_token."
            )

        input_tokens = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        input_details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
        cached_tokens = int(input_details.get("cached_tokens") or 0) if isinstance(input_details, dict) else 0
        uncached_tokens = max(0, input_tokens - cached_tokens)

        cost = uncached_tokens * input_rate + output_tokens * output_rate
        if cached_tokens:
            cost += cached_tokens * (cached_rate if cached_rate is not None else input_rate)
        return {"cost": float(cost)}

    def format_observation_messages(
        self,
        message: dict[str, Any],
        outputs: list[dict[str, Any]],
        template_vars: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        actions = message.get("extra", {}).get("actions", [])
        return format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return {
            **self.config.model_dump(),
            "openai_api_model_name": self.api_model_name,
            "openai_previous_response_id": self._previous_response_id,
        }


def finish_reason_from_responses_api(response: Any) -> str | None:
    if hasattr(response, "status"):
        return getattr(response, "status", None)
    if isinstance(response, dict):
        return response.get("status")
    return None


class KindleStatefulResponsesModel(OpenAIResponsesCoTModel):
    """Compatibility alias for Pier configs targeting Kindle alpha."""
