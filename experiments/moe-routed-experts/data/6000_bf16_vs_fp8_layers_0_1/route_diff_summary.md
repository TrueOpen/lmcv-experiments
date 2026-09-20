# Routed Experts Diff Summary

- samples: 1000
- ok samples: 1000
- selected layers: ['0,1']
- selected layer counts: [2]
- weighted entry mismatch rate: 0.1317805745210976
- weighted token match rate: 0.3086412762101993
- weighted token-layer set match rate: 0.8620526469065494
- mean top-k overlap: 0.9821797585087284
- mean top-k jaccard: 0.9684759739597101
- total compared entries: 1977856
- total compared token-layers: 247232
- shape mismatch samples: 0
- raw shape mismatch samples: 1000
- expected length mismatch samples: 0
- layer/top-k shape mismatch samples: 0

## Sample Metric Summary

| metric | count | mean | min | p50 | p90 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| entry_mismatch_rate | 1000 | 0.13166464891311574 | 0.0 | 0.1235236220472441 | 0.1451771653543307 | 0.3764763779527559 |
| token_layer_match_rate | 1000 | 0.5464541895138498 | 0.0 | 0.5590551181102362 | 0.6063174394867309 | 1.0 |
| token_layer_set_match_rate | 1000 | 0.8608855598528263 | 0.0 | 0.8740157480314961 | 0.905511811023622 | 1.0 |
| token_match_rate | 1000 | 0.309086497980371 | 0.0 | 0.31496062992125984 | 0.3779623583637411 | 1.0 |
| topk_overlap_mean | 1000 | 0.9821797585087284 | 0.875 | 0.984251968503937 | 0.9876968503937008 | 1.0 |
| topk_jaccard_mean | 1000 | 0.9684759739597101 | 0.7777777777777778 | 0.9720034995625548 | 0.97830271216098 | 1.0 |
