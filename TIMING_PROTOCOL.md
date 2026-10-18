# Repeated timing validation

Fixed before inspecting confirmation results. After confirmation and the
additional-domain evaluation, load baseline6 and the locked winner from seed
777, the preselected last confirmation seed. Use fresh optimizer/scaler states
and fixed data seed 888. This timing-only test does not produce quality scores
and does not modify the saved model files.

Use the same default full-graph compiled FP16/FP32 training path, batch 32 and
sequence 128. Warm up each model for eight updates, then measure five repetitions
of 50 updates per model. Reverse model execution order in alternate repetitions.
Use matching training blocks within each repetition. AdamW fused LR 1e-4,
betas (.9,.95), weight decay .1, gradient norm clip 1, loss scale initially 1,024.
Include batch construction, transfer and complete synchronized optimizer work.

Report per-repetition median throughput, paired speed change, overall step
timings and source checkpoint hashes. Do not replace an unfavorable primary
confirmation criterion with this additional result. This check reduces time
separation between comparisons, but shared hardware remains a limitation.
