# Routed Experts Diff Summary

- samples: 1000
- ok samples: 1000
- selected layers: ['0,1,2,3,4']
- selected layer counts: [5]
- weighted entry mismatch rate: 0.16938664897747865
- weighted token match rate: 0.029203339373543878
- weighted token-layer set match rate: 0.8110940331348693
- mean top-k overlap: 0.9751514160411621
- mean top-k jaccard: 0.956261952486012
- total compared entries: 4944640
- total compared token-layers: 618080
- shape mismatch samples: 0
- raw shape mismatch samples: 1000
- expected length mismatch samples: 0
- layer/top-k shape mismatch samples: 0

## Sample Metric Summary

| metric | count | mean | min | p50 | p90 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| entry_mismatch_rate | 1000 | 0.16926998040328134 | 0.12421875 | 0.15846456692913385 | 0.1773005565862709 | 0.4828740157480315 |
| token_layer_match_rate | 1000 | 0.4571713673568482 | 0.06141732283464567 | 0.47086614173228347 | 0.5070866141732283 | 0.56875 |
| token_layer_set_match_rate | 1000 | 0.8105916383768498 | 0.2 | 0.8267716535433071 | 0.8500393700787402 | 1.0 |
| token_match_rate | 1000 | 0.029049730535182564 | 0.0 | 0.03125 | 0.05511811023622047 | 0.09448818897637795 |
| topk_overlap_mean | 1000 | 0.9751514160411621 | 0.875 | 0.977755905511811 | 0.9809055118110236 | 1.0 |
| topk_jaccard_mean | 1000 | 0.956261952486012 | 0.7866666666666666 | 0.9607349081364829 | 0.9661242344706911 | 1.0 |
