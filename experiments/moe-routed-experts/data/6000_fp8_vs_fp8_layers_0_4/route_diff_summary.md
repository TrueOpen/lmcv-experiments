# Routed Experts Diff Summary

- samples: 1000
- ok samples: 1000
- selected layers: ['0,1,2,3,4']
- selected layer counts: [5]
- weighted entry mismatch rate: 0.12623347750515382
- weighted token match rate: 0.07556489753021545
- weighted token-layer set match rate: 0.8592182384089898
- mean top-k overlap: 0.9819004944838688
- mean top-k jaccard: 0.9680109941250757
- total compared entries: 4947800
- total compared token-layers: 618475
- shape mismatch samples: 0
- raw shape mismatch samples: 1000
- expected length mismatch samples: 0
- layer/top-k shape mismatch samples: 0

## Sample Metric Summary

| metric | count | mean | min | p50 | p90 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| entry_mismatch_rate | 1000 | 0.1260304449231676 | 0.05 | 0.1188976377952756 | 0.13446850393700788 | 0.397244094488189 |
| token_layer_match_rate | 1000 | 0.5706867860807749 | 0.1732283464566929 | 0.5811023622047244 | 0.6157641374373658 | 0.8 |
| token_layer_set_match_rate | 1000 | 0.8592833366325874 | 0.49921259842519683 | 0.8692913385826772 | 0.889763779527559 | 1.0 |
| token_match_rate | 1000 | 0.0754381374822637 | 0.0 | 0.07874015748031496 | 0.11811023622047244 | 0.1782178217821782 |
| topk_overlap_mean | 1000 | 0.9819004944838688 | 0.9175196850393701 | 0.9832677165354331 | 0.9860236220472441 | 1.0 |
| topk_jaccard_mean | 1000 | 0.9680109941250757 | 0.862763797881908 | 0.9703937007874016 | 0.9752230971128608 | 1.0 |
