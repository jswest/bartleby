# Never send temperature to the OpenAI provider (issue #222)

> Source: [#222](https://github.com/jswest/bartleby/issues/222)

The OpenAI provider used to forward the user's configured `temperature` to
`chat.completions.parse`. But the **GPT-5 family** (`gpt-5`, `gpt-5-mini`,
`gpt-5-nano`) — the only models bartleby pairs with this provider — accepts
**solely the API default of `1.0`**; any other value is rejected outright
(`Unsupported value: 'temperature' does not support 0.0 with this model. Only
the default (1) value is supported.`). With the stock config (default model
`gpt-5-mini`, `DEFAULT_TEMPERATURE = 0.0`) this meant **every** summarize/classify
call errored out of the box.

**The fix is to omit the parameter entirely, not to clamp it to a value.** We
never send `temperature` to OpenAI and let the API default stand. Two reasons to
omit rather than hardcode `1.0`: (1) it's robust if OpenAI ever shifts the
default, and (2) the discriminator is the *provider*, not the model string — the
OpenAI provider only ever drives GPT-5 models here, so a per-model prefix check
(`startswith("gpt-5")`) would be needless machinery for a distinction the
provider boundary already makes. If a non-GPT-5 OpenAI model that honors
`temperature` ever enters the picture, that's the moment to reintroduce a model
check — not before.

Dropping a configured value silently would mislead (a user sets `temperature:
0.2`, sees no effect, and has no idea why), so the provider **warns once per
process** when it drops a non-`1.0` value, via `console.warn` — which shares the
scribe `Live` console so the notice inserts above the progress bar instead of
stomping it, matching the `_warn_ocr_degraded_once` precedent in
`bartleby/ingest/images.py`. Once-per-process, not once-per-call: summarize runs
per-document, so a per-call warning would emit thousands of lines on a large
ingest. `analyze_image` already omitted `temperature` and is unchanged. No schema
change — patch-level.
