# Phase-three protocol, fixed before new training

Date: 18 October 2026. Aim: reduce the Block AttnRes cost and investigate a
smaller model, with a fair optimized baseline and explicitly defined quality.

## Candidates and selection

Six candidates: ordinary six-layer and five-layer MiniMind; tuple-based AttnRes
with six/five layers and block sizes four/twelve sublayers. All use hidden width
384, MiniMind's original tokenizer, sequence 128, FP32 residual and block states,
FP16 linear autocast, fused AdamW and the same compilation settings. Block twelve
can exceed the ten sublayers of the five-layer model; its final mixer includes
the unfinished partial block. This is a documented configuration, not a full K3
reproduction. Smaller models have fewer parameters and are explicitly labeled.

The tuple implementation avoids stacking full hidden-state sources and uses the
two-source sigmoid identity. Reference outputs/gradients, causal behavior,
shared initialization and mixed-precision states must pass correctness checks.
Compiled/eager gradients use the trainer's loss scaling and unchanged tolerance.

Seed 42 is for development only: all six candidates train from scratch for
1,000 updates (4,096,000 predicted tokens each). This uses the existing train and
development-validation caches. No new heldout scores are computed during this
screen. Among AttnRes candidates whose endpoint development NLL is at least
0.005 below baseline6, select the highest measured steady throughput. If none
qualifies, select the lowest development NLL and record that the quality
selection condition failed. Do not select using the confirmation holdout.

## Locked confirmation

After selection, lock its configuration. Run baseline6, baseline5 and the winner
at fresh confirmation seeds 17, 123 and 777. Seeds 17/123 appeared in phase two
but are unused in this phase's selection; 777 is new to the study. The primary
three-seed confirmation is independent of the seed-42 development screen.
Use rotated execution order across the three seeds. All runs receive matching
initial shared weights at a given depth and matching sampled training blocks.

The fixed budget is 1,000 updates, physical/logical batch 32, sequence 128,
AdamW fused, LR 3e-4, 50-update warmup/cosine decay, betas (.9,.95), weight decay
.1, gradient clipping 1.0, initial loss scale 1,024. No skipped nonfinite steps.
Evaluate development NLL every 50 updates. These settings stay unchanged for
all configurations; a new experiment is needed for a new setting.

Prepare a new 128,000-token partition from subsequent TinyStories validation
stories, excluding complete stories used in training, earlier validation and
phase-two holdout by exact text hash. Reconstruct and verify all preceding
token caches. The new partition remains part of the official validation split,
not an official test split or independent domain. Evaluate it only at each
confirmation run's fixed endpoint; never use it to choose the winner.

## Speed and success criteria

- Raw training throughput: median synchronized predicted tokens/s after eight
  warmup updates, excluding evaluation/checkpoint IO. Use the same engine for all.
- Fixed quality target: development NLL <= 2.10. Record the first observed
  crossing at 50-update evaluation intervals, without interpolation. Report both
  cumulative training-step seconds and wall seconds including model setup,
  compiler-cache loading/compilation, evaluation and runtime checkpoint writes.
  Data/dependency preparation and final Drive backup are outside this timing.
- Steady speed is not cold-start or cached decoding speed. Record first-step
  time, parameter count, peak allocated memory and compiler-cache limitations.
- A local faster-and-stronger result requires the selected model to have lower
  confirmation-holdout NLL and higher raw throughput than baseline6 in every
  confirmation seed. A faster time to target is a separate claim and cannot
  substitute for a failed raw-throughput comparison. Report means, sample SD,
  individual pairs and any failures, without a general-capability claim.

## Preservation

Keep earlier sources and scores intact. Save full resumable checkpoints in the
runtime every 250 updates. Back up new completed model weights, configuration,
logs and scores to the approved Drive folder, using SHA-256 verification. To
conserve Drive space, these new persistent model files omit optimizer/scaler
state; they support inference/evaluation but cannot resume the exact optimizer
trajectory after runtime loss. Retain that limitation in the report.

