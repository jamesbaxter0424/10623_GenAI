# NYC LLM request generation and parsing

Qwen3-8B (few-shot/prompt-based, no fine-tuning) pilot on
10 real NYC training blocks.

## Request types
- Exact numeric
- Natural numeric
- Qualitative FAR category (NYC city-wide quantiles)
- Neighbor-comparative FAR relation

## Results
See `results/evaluation_summary.csv`.

40 generated requests passed their style-specific
round-trip checks in this preliminary, same-model pilot.

Numeric requests recover block ID, coverage, FAR,
height class and residential share with fixed tolerances.
Qualitative requests recover FAR category, and comparative
requests recover relative FAR relationships; these two
do not reconstruct a complete numeric block table.

## Data
The complete processed NYC dataset is not committed.
Pilot ground truth and generated request results are in
`results/`.

## Limitations
- The same Qwen3-8B model generated and parsed requests
  using separate prompts.
- The pilot uses 10 selected training blocks, not an
  independent test set.
- Multi-block requests and independent human-authored
  requests have not yet been evaluated.
