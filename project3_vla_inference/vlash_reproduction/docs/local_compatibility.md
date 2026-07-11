# Local Compatibility Patch

The upstream Pi0.5 policy loads `google/paligemma-3b-pt-224` by repository
name. The AutoDL experiment host keeps the tokenizer in a local directory, so
offline runs need a path override.

`0001-local-paligemma-tokenizer.patch` preserves the upstream repository name
as the default and introduces one opt-in environment variable:

```text
VLASH_PALIGEMMA_TOKENIZER_PATH=/root/autodl-tmp/vla-infra-project3-pi05/models/paligemma-3b-pt-224-tokenizer
```

This patch does not alter the Pi0.5 model, LoRA, delay augmentation, shared
observation forward path, or asynchronous runtime algorithm.
