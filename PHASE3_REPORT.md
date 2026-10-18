# mini-k3: lower-cost attention residuals and compact-model confirmation

Date: 18 October 2026. Author: Fazal Hameed.

## Outcome

Selected model: **attnres5_b12**, 11,441,152 parameters (13.51% fewer than baseline6). The mean paired heldout-perplexity reduction was **3.93%** and the mean paired raw-throughput change was **+12.56%**. The internally fixed faster-and-stronger criterion was **MET within the stated small-model TinyStories comparison**.

The additional custom WikiText-2 test-prefix perplexity change was a +5.49% reduction (positive means better). Lower WikiText NLL in every seed: False. These results do not establish general LLM capability, Chinese ability or assistant quality.

## Experiment design

The six-layer pilot MiniMind baseline and five-layer ordinary-residual control were compared with four AttnRes candidates: five/six layers and blocks of four/twelve sublayers. All retain width 384, six query/two KV heads, vocabulary 6,400, the original tokenizer, causal grouped-query attention, rotary positions, dense feed-forward layers and FP32 residual/block accumulation under FP16 linear autocast. The models train from scratch. MiniMind is pinned at `f659b55761b754d306bd140573493a6543cafd7f` (Apache-2.0).

Block Attention Residuals follows the depth-mixing idea in [Kimi K3, section 2.2](https://arxiv.org/html/2607.24653v2#S2.SS2). The new implementation stores completed block states in tuples rather than repeatedly stacking full hidden-state tensors. It caches each completed state’s inverse RMS within the differentiable forward computation. With two source scores s0/s1 and values v0/v1, the second softmax weight equals sigmoid(s1-s0); the mixture is v0 + gate*(v1-v0). More than two sources use small score stacks and weighted sums. The sigmoid identity is algebraically equivalent, with floating-point roundoff differences.

Larger blocks reduce the number of completed depth sources, while the mixer is still evaluated before each attention/MLP and at the output. A block size of twelve exceeds a five-layer model’s ten sublayers; its final mixer includes the unfinished partial sum. Compactness is a real architecture change, and the comparison is not parameter-matched. The five-layer ordinary control separates depth reduction from adding AttnRes. Matching-depth models share identical initial weights; different-depth models use the same seed but do not assert all initial tensors are identical. No KDA, LatentMoE, SiTU-GLU or Muon is implemented. No KV-cache or decoding-speed claim is made.

## Development screen and selection

An internal protocol was written before the new training. Seed 42 was used only for development. Each of six candidates trained for 1,000 updates. Among AttnRes candidates whose endpoint development NLL beat baseline6 by at least 0.005, the highest measured steady training throughput was selected. If no candidate passed, the lowest-NLL AttnRes candidate would be selected and the condition flagged as failed. Selection never used new confirmation holdout or WikiText scores. This was an internal prospective plan, not an externally registered study.

| Candidate | Parameters | Development NLL | Training tokens/s | First step, s |
| --- | ---: | ---: | ---: | ---: |
| baseline6 | 13,227,648 | 2.0127 | 136712 | 4.69 |
| baseline5 | 11,432,704 | 2.0217 | 152977 | 14.57 |
| attnres6_b4 | 13,237,632 | 1.9432 | 111054 | 29.41 |
| attnres6_b12 | 13,237,632 | 1.9511 | 110710 | 23.85 |
| attnres5_b4 | 11,441,152 | 1.9663 | 124449 | 24.18 |
| attnres5_b12 | 11,441,152 | 1.9736 | 131852 | 19.12 |

Selected: `attnres5_b12`. Quality selection condition passed: True.

## Locked training and heldout evaluation

- Three confirmation seeds: 17, 123 and 777. Seeds 17/123 occurred in earlier work, but were not used to select this phase’s winner; 777 is new. The development screen uses seed 42 only.
- Baseline6, baseline5 and the locked winner run at every seed, with execution order rotated.
- Free Tesla T4 (14.56 GiB), Python 3.13.15, PyTorch 2.11.0+cu130, Transformers 4.57.6.
- Physical/logical batch 32, 128 predicted tokens per sequence: 4,096 predicted tokens per update.
- 1,000 updates and 4,096,000 processed training tokens per run. Same seed/step-dependent batch stream.
- Fused AdamW, peak LR 3e-4, 50-update warmup and cosine decay toward 10% peak, betas (.9,.95), weight decay .1.
- FP32 parameters and residual/block states; FP16 linear autocast; initial loss scale 1,024; clip norm 1.0.
- Full-graph `torch.compile`, default mode, fixed shapes. All candidates pass compiled/eager scaled-gradient checks.
- Development NLL evaluated every 50 updates on the same 32,768 predicted tokens. No hyperparameter sweep.
- Nonfinite loss/gradients halt training; no silently skipped updates or unequal token budgets.

Training uses the same pinned, exact-text-deduplicated two-million-token TinyStories cache. It is sampled with replacement, so processed-token count is not a distinct-token count. Dataset revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, CDLA-Sharing-1.0. The training cache, original validation prefix and phase-two heldout prefix were reconstructed and checked for exact token equality. A new partition uses 492 subsequent complete validation stories, 128,000 cached tokens and 126,976 predicted tokens in full packed blocks. Complete selected train, original validation and earlier heldout stories are excluded by exact text hash. Near-duplicate/paraphrase removal was not performed. The new partition is not an official test split or a new domain; it is evaluated only at the fixed confirmation endpoint.

Perplexity is exp(mean next-token NLL in nats/token). Packed blocks can cross story boundaries and do not mask attention at EOS. Both models use the same packing, tokenizer and causal mask.

## Three-seed confirmation results

| Seed | Candidate | Heldout NLL | Heldout perplexity | Training tokens/s | Target step | Target wall, s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 17 | baseline6 | 2.2870 | 9.846 | 117659 | 700 | 32.90 |
| 17 | baseline5 | 2.3010 | 9.985 | 144089 | 750 | 25.67 |
| 17 | attnres5_b12 | 2.2483 | 9.472 | 134195 | 650 | 24.43 |
| 123 | baseline5 | 2.2870 | 9.846 | 141461 | 750 | 24.90 |
| 123 | attnres5_b12 | 2.2387 | 9.381 | 135214 | 650 | 23.07 |
| 123 | baseline6 | 2.2879 | 9.854 | 120753 | 700 | 27.26 |
| 777 | attnres5_b12 | 2.2378 | 9.373 | 134859 | 650 | 23.17 |
| 777 | baseline6 | 2.2702 | 9.681 | 120793 | 700 | 27.17 |
| 777 | baseline5 | 2.2935 | 9.910 | 140794 | 750 | 24.85 |

Means ± sample SD across three confirmation seeds:

| Candidate | Heldout perplexity | Training tokens/s | Target wall, s | Peak GPU MiB |
| --- | ---: | ---: | ---: | ---: |
| baseline6 | 9.794 ± 0.098 | 119735 ± 1798 | 29.11 ± 3.28 | 638.0 ± 0.6 |
| baseline5 | 9.913 ± 0.070 | 142115 ± 1742 | 25.14 ± 0.46 | 551.5 ± 1.3 |
| attnres5_b12 | 9.409 ± 0.055 | 134756 ± 517 | 23.56 ± 0.76 | 550.7 ± 1.1 |

Paired winner changes against baseline6 (positive percentages mean improvement):

| Seed | Heldout perplexity reduction | Throughput increase | Target wall reduction | WikiText perplexity reduction |
| --- | ---: | ---: | ---: | ---: |
| 17 | +3.80% | +14.05% | +25.74% | +14.96% |
| 123 | +4.80% | +11.98% | +15.38% | -1.05% |
| 777 | +3.18% | +11.65% | +14.72% | +2.57% |

## Repeated timing validation

A separate timing-only check loads the fixed seed-777 confirmation checkpoints, uses fresh AdamW/scaler states, warms up eight updates per model, then measures five repetitions of 50 updates each. The execution order alternates. Data seed 888, LR 1e-4 and the same precision/compiler settings are fixed. The temporary benchmark updates do not modify saved weights and do not produce quality scores. This check reduces the separation between speed comparisons; it does not replace the primary criterion.

| Repetition | Baseline6 tokens/s | Winner tokens/s | Winner speed change |
| --- | ---: | ---: | ---: |
| 1 | 122804 | 134971 | +9.91% |
| 2 | 119138 | 134370 | +12.78% |
| 3 | 117089 | 127836 | +9.18% |
| 4 | 112933 | 127398 | +12.81% |
| 5 | 111817 | 123223 | +10.20% |

Mean paired repeated-benchmark speed change: +10.98%. Higher winner throughput in all five repetitions: True.

## Time to a fixed prediction quality

The fixed target is development NLL <= 2.10, chosen from the earlier phase before these runs. The first observed crossing is measured at 50-update intervals; it is not interpolated or an exact earliest crossing. Cumulative training-step time and wall time are both retained. Wall time includes model creation, compiler-cache loading/compilation, development evaluation and any runtime checkpoints written before the crossing. It excludes dependency/data preparation and final model export/Drive backup. No Drive checkpoint writes occur inside the timed training run. These are training-time measurements and cannot be described as inference latency.

Mean paired target wall-time reduction: +18.61%.
Faster time to this one quality threshold is a separate metric. It cannot replace a failed raw throughput result or establish performance at every quality target.

Steady throughput includes synchronized batch construction/transfer, forward/backward, scaling, clipping and optimizer work after the first eight updates; it excludes evaluation/checkpoint IO. Compiler caches were primed by correctness checks and development runs. First-step times are recorded but are not a controlled empty-cache compilation experiment. Shared free hardware, thermal/load variation, a single session and limited order rotation restrict small timing differences. Peak memory is PyTorch allocated memory, not reserved or total-device use.

## Additional-domain check

After selection and confirmation training, all nine checkpoints are evaluated on the same custom [WikiText-2 raw test prefix](https://huggingface.co/datasets/Salesforce/wikitext), pinned at `b08601e04326c79dfdd32d625aee71d232d685c3`. The unchanged MiniMind tokenizer encodes stripped nonempty lines with EOS after each; the first 128,000 tokens are packed as above. This evaluates 126,976 predicted tokens. License tags on the source card are CC-BY-SA-3.0 and GFDL. The additional-domain plan was fixed before selection and inspecting development endpoint results. This is not the standard published WikiText perplexity protocol and is not comparable with numbers using another tokenizer, context or line handling. No WikiText training or tuning was performed.

| Seed | Candidate | Custom WikiText NLL | Custom WikiText perplexity |
| --- | --- | ---: | ---: |
| 17 | baseline6 | 7.6952 | 2197.713 |
| 17 | baseline5 | 7.7059 | 2221.451 |
| 17 | attnres5_b12 | 7.5331 | 1868.982 |
| 123 | baseline5 | 7.5654 | 1930.184 |
| 123 | attnres5_b12 | 7.5895 | 1977.401 |
| 123 | baseline6 | 7.5790 | 1956.758 |
| 777 | attnres5_b12 | 7.5072 | 1821.089 |
| 777 | baseline6 | 7.5332 | 1869.086 |
| 777 | baseline5 | 7.5553 | 1910.856 |

## Interpretation and limits

Primary condition: lower TinyStories heldout NLL in every seed = True; higher throughput in every seed = True. Additional-domain lower NLL in every seed = False. A local success supports only this compact-model, fixed-budget comparison on the measured hardware. It does not show that adding AttnRes to an otherwise identical model has zero cost. The same-depth ordinary control and development candidates reveal that distinction. A larger model study, more domains, downstream tasks, cached inference and repeated timing sessions remain. Three seeds are not proof of general superiority.

The custom WikiText perplexities are very high in absolute terms: all these story-trained models remain poor encyclopedic language models. A mixed additional-domain result prevents a claim of consistent superiority across domains. Phase-three and phase-two heldout sets differ, so their percentage improvements should not be compared as if measured on identical text.

Phase one’s original paths accumulated residuals at different precision. Phase two corrected that confound and found a quality gain with a speed cost. Phase three preserves the precision control, changes the implementation/block size, and explicitly tests smaller capacity. Earlier sources and measurements remain intact; the new result must not be substituted retrospectively into old runs.

## Correctness and artifacts

CPU/reference checks cover tuple/reference logits and gradients with nonzero queries, causality, matching-depth initial weights, and FP32 states under GPU autocast. All six candidates passed compiled/eager losses and scaled-gradient checks at batch 32/context 128. These correctness tests are distinct from model-quality results. Configuration, data/code hashes, histories, per-step timing and individual confirmation scores are retained. No GPU interrupted/resumed trajectory-equivalence test was performed for this runner.

Full resumable checkpoints were written every 250 updates in the original Colab runtime. That runtime expired after the experiments finished; its temporary optimizer checkpoints were lost. New completed model-only files, configs, results and logs are backed up with SHA-256 verification under `My Drive/mini-k3/2026-10-18/phase3`. These persistent model exports omit AdamW/scaler state and cannot resume the exact optimizer trajectory after runtime loss. They can reproduce evaluation with the pinned source. Local bundles contain code, measured evidence and reports; model and cached-token bytes remain in Drive/the runtime.

