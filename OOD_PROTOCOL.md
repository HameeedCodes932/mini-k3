# Additional-domain evaluation, fixed before development results

18 October 2026. This addendum is fixed while the first development model is
still running, before any candidate is selected or any additional-domain score
is computed. It does not change the candidate-selection rule or primary criteria
in PROTOCOL.md.

After the selection is locked and confirmation training has finished, evaluate
the nine confirmation checkpoints on a fixed 128,000-token prefix of
`Salesforce/wikitext`, configuration `wikitext-2-raw-v1`, official `test` split,
revision `b08601e04326c79dfdd32d625aee71d232d685c3`.

Use the unchanged MiniMind tokenizer. Strip each source line, skip empty lines,
append EOS after each nonempty line, and concatenate tokens in dataset order.
Use non-overlapping packed 129-token blocks, predicting 128 tokens per block.
Do not fine-tune, select candidates, alter budgets or retry new architectures
using this score. Report the same NLL and perplexity metrics for every model.

This is a custom tokenized test-prefix evaluation, not the published standard
WikiText-2 perplexity protocol. It provides a limited domain-shift check on
encyclopedic English after training on simple stories. Tokenizer, line-level EOS
and context packing restrict external comparisons. Prefix selection need not
represent the full dataset. No general reasoning or Chinese-language claim is
supported. Report any regression even if the primary TinyStories result wins.

Sources: https://huggingface.co/datasets/Salesforce/wikitext and the pinned
commit at https://huggingface.co/datasets/Salesforce/wikitext/commit/b08601e04326c79dfdd32d625aee71d232d685c3.
Dataset card license tags: CC-BY-SA-3.0 and GFDL. Keep attribution with any
redistributed token cache. The local result bundle excludes cached token bytes.
