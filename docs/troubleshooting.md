# Troubleshooting Notes

## Conda Solver Broken on AutoDL Image

Some AutoDL images had Python 3.12 and a broken conda libmamba solver. The practical fix was to choose an image with Python 3.10 and PyTorch 2.1.2+cu118 instead of fighting the base image.

## Python Version

Nanotron is more reliable with Python 3.10 for this project. Python 3.12 caused compatibility friction with package metadata and CUDA extension packages.

## grouped_gemm Missing `pkg_resources`

Error:

```text
ModuleNotFoundError: No module named 'pkg_resources'
```

Fix:

```bash
python -m pip install -U "setuptools<70" wheel packaging ninja
```

## pytest `-n` Argument Unknown

Error:

```text
pytest: error: unrecognized arguments: -n
```

Cause: Nanotron test config expects `pytest-xdist`.

Fix:

```bash
pip install pytest-xdist
```

## flash-attn Missing

Error:

```text
ModuleNotFoundError: No module named 'flash_attn'
```

Fix: install flash-attn compatible with the active PyTorch/CUDA environment.

## flash-attn Too Old for Fused LayerNorm

Error:

```text
ModuleNotFoundError: No module named 'flash_attn.ops.triton.layer_norm'
```

Fix:

```bash
pip uninstall -y flash-attn
MAX_JOBS=4 pip install --no-build-isolation --no-cache-dir --force-reinstall --no-deps "flash-attn==2.5.8"
```

## `torch.utils.collect_env` Attribute Error

Error:

```text
AttributeError: module 'torch.utils' has no attribute 'collect_env'
```

Cause: PyTorch 2.1.2 does not expose `collect_env` through `torch.utils` unless explicitly imported.

Patch:

```python
import torch
import torch.utils.collect_env
```

in `src/nanotron/logging/base.py`.

## Dummy-Data Resume Metadata Assertion

Error:

```text
AssertionError: Mismatch between the total consumed tokens and the sum of consumed tokens across stages! Something went wrong in the training.
```

Cause: the dummy CLM dataloader has no dataset folder. Nanotron still increments the global `consumed_tokens_total`, but `consumed_tokens_per_dataset_folder` can remain empty. On resume, strict metadata sanity checking compares the global token count against the sum of per-dataset token counts and fails.

Temporary smoke-test patch:

```python
stage_consumed_tokens_total = sum(stage.consumed_tokens_all_datasets for stage in self.data_stages)
if self.consumed_tokens_total is not None:
    if stage_consumed_tokens_total != 0:
        assert self.consumed_tokens_total == stage_consumed_tokens_total
else:
    self.consumed_tokens_total = stage_consumed_tokens_total
```

This keeps the strict check for real datasets while allowing dummy-data checkpoint resume validation.
