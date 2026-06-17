# mini-swe-agent-openai-cot

Stateful OpenAI Responses API model adapter for `mini-swe-agent`.

The built-in `litellm_response` model flattens previous Responses API outputs
back into the next request but does not also set `previous_response_id`. This
adapter stores the last OpenAI response ID and sends every turn with the full
Responses API history plus `previous_response_id`. It always sends:

```python
reasoning={"context": "all_turns"}
include=["reasoning.encrypted_content"]
store=True
```

## Pier config

```yaml
agents:
  - name: mini-swe-agent
    model_name: openai/kindle-alpha-api
    env:
      OPENAI_API_KEY: ${OPENAI_API_KEY}
    kwargs:
      model_class: mswea_openai_cot.KindleStatefulResponsesModel
      extra_python_packages:
        - "mini-swe-agent-openai-cot @ git+https://github.com/YOUR_ORG/mini-swe-agent-openai-cot.git@PINNED_SHA"
      model_kwargs:
        # Adapter settings. Pier forwards these under mini-swe-agent's
        # model.model_kwargs, so the adapter strips them before the API call.
        api_model_name: kindle-alpha-api
        reasoning:
          effort: xhigh
          summary: auto
        log_raw_requests: true
```

Any normal Responses API parameter that is not owned by the adapter can go under
`model_kwargs`.

## How it works

This package is meant to be installed into the Python environment that already
contains `mini-swe-agent`. It intentionally does not declare `mini-swe-agent` as
a dependency, so installing the adapter will not upgrade or replace the runner.

On the first call the adapter sends the full initial mini-swe-agent input. After
OpenAI returns `resp_...`, the adapter stores that ID. On later calls, the
adapter still sends the full local trajectory, including prior response output
items, and also passes:

```python
previous_response_id="resp_..."
```

This matches the pattern:

```python
history.extend(item.model_dump(exclude_none=True) for item in response.output)
client.responses.create(previous_response_id=response.id, input=history, ...)
```

Both first and later calls use `reasoning.context="all_turns"`.

## Smoke check

After installing through Pier, inspect `agent/mini-swe-agent.trajectory.json`.
You should see:

- OpenAI response objects with `id: resp_...`.
- Later response objects with `previous_response_id` set.
- Reasoning output items containing `encrypted_content` when the API returns it.
- Prior response output items present again in later request inputs.
- The exact Responses API request kwargs under `extra.openai_cot.request`.

If `log_raw_requests: true` is set, the same request kwargs are also written to
`agent/mini-swe-agent.txt` before each API call. These logs include prompts and
encrypted reasoning items, so treat rollout artifacts as sensitive.
