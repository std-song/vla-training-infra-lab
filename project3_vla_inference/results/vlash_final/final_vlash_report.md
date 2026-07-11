# Final VLASH Pi0.5 Reproduction

## Scope

This reproduction uses upstream VLASH training and `VLASHAsyncManager` with a
local Pi0.5 base model, LeRobot ALOHA multi-camera data, TorchCodec, LoRA, and
the official future-state/shared-observation configuration. A small offline
robot adapter supplies recorded observations; it does not replace VLASH
scheduling or future-state logic.

## Final Fine-tuning

| Item | Value |
| --- | ---: |
| Dataset | `lerobot/aloha_mobile_cabinet`, 85 episodes / 127,500 frames |
| Policy | Pi0.5, 3.77B total / 154M trainable LoRA parameters |
| Steps | 1,000 |
| Delay training | offsets 0..8, shared observation enabled |
| Final loss | 0.059 |
| Stable update time | 0.63-0.78 s |
| Stable data time | about 1 ms |
| Checkpoint | step 1,000, policy 7.48 GiB + optimizer 1.09 GiB |

With `shared_observation=true`, VLASH constructs every offset 0..8 for a
sample and shares visual/language observation embeddings. This is distinct
from the non-shared dataset path, which randomly samples one delay offset.
The current upstream default uses the preceding action as a future-state proxy
rather than recorded future state.

## Replay Runtime

The final checkpoint was replayed for 96 recorded ALOHA observations through
the upstream manager in three configurations: sync (`overlap=0`), VLASH async
(`overlap=4`), and async plus action quantization ratio 2.

| Measurement | Sync | Async | Async q2 |
| --- | ---: | ---: | ---: |
| First action chunk | 91.6 s | 94.2 s | 97.9 s |
| Queue-pop calls | about 0.04-0.10 ms | about 0.04-0.10 ms | about 0.05-0.10 ms |
| Replay rows | 96 | 96 | 96 |

These offline traces validate checkpoint loading, action-chunk generation,
future-state scheduling, and quantized action-send cadence. They do not prove
robot-side speedup: the public manager performs policy generation inline and
the replay has no physical actuator, communication, or 30 Hz wall-clock loop.
VLASH async/quantization should therefore be evaluated on control-tick stalls,
action age, and robot command overhead when hardware is available, not with
mean `get_action` wall time alone.
