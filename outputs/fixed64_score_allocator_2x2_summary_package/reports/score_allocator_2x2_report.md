# 2×2 Score × Allocator Attribution Audit

## Scope and hard constraints

- No training, backward pass, optimizer step, MLP forward pass, checkpoint selection, factor selection, direction change, temperature tuning, cap, sparse allocator, Factor Scale Ablation, or N1 run.
- All four corners use the existing 2022-01-04 to 2026-09-01 OOS score axes and the same drift-aware continuous accounting, CSI500 path, and 7.5bp one-side cost convention.

## Core 2×2 stitched table

| score_type   |   annual_return_0p0bps|Softmax T=1 |   annual_return_0p0bps|Top20 EW |   annual_return_7p5bps|Softmax T=1 |   annual_return_7p5bps|Top20 EW |   sharpe_7p5bps|Softmax T=1 |   sharpe_7p5bps|Top20 EW |   turnover_mean|Softmax T=1 |   turnover_mean|Top20 EW |   effective_n|Softmax T=1 |   effective_n|Top20 EW |   information_ratio|Softmax T=1 |   information_ratio|Top20 EW |
|:-------------|-----------------------------------:|--------------------------------:|-----------------------------------:|--------------------------------:|----------------------------:|-------------------------:|----------------------------:|-------------------------:|--------------------------:|-----------------------:|--------------------------------:|-----------------------------:|
| Factor Mean  |                           0.054912 |                        0.058458 |                           0.037923 |                        0.023158 |                    0.302396 |                 0.219837 |                    0.042967 |                 0.089766 |                319.095338 |              99.740708 |                        0.248877 |                     0.049756 |
| MLP          |                           0.043648 |                        0.051786 |                           0.015552 |                       -0.004105 |                    0.175011 |                 0.074874 |                    0.072213 |                 0.144474 |                299.924891 |              99.740708 |                        0.023452 |                    -0.216829 |

## Existing-corner identity

- FM-Top20: **PASS**; MLP-Softmax: **PASS**.
- Eligible/valid/execution mask: **PASS**; RankIC allocator invariance: **PASS**.

## Stitched attribution effects

| effect                          |   delta_annual_return_0p0bps |   delta_annual_return_7p5bps |   delta_sharpe_7p5bps |   delta_turnover_mean |   delta_transaction_cost_drag_annual |   delta_effective_n |   delta_information_ratio |
|:--------------------------------|-----------------------------:|-----------------------------:|----------------------:|----------------------:|-------------------------------------:|--------------------:|--------------------------:|
| allocator_effect_on_factor_mean |                    -0.003546 |                     0.014766 |              0.082558 |             -0.046799 |                            -0.018311 |          219.354630 |                  0.199121 |
| scorer_effect_under_top20       |                    -0.006671 |                    -0.027262 |             -0.144963 |              0.054708 |                             0.020591 |            0.000000 |                 -0.266585 |
| scorer_effect_under_softmax     |                    -0.011264 |                    -0.022372 |             -0.127385 |              0.029246 |                             0.011108 |          -19.170446 |                 -0.225425 |

## 2×2 interaction

| definition                                          |   interaction_annual_return_0p0bps |   interaction_annual_return_7p5bps |   interaction_sharpe_7p5bps |   interaction_turnover_mean |
|:----------------------------------------------------|-----------------------------------:|-----------------------------------:|----------------------------:|----------------------------:|
| (MLP-Softmax - MLP-Top20) - (FM-Softmax - FM-Top20) |                          -0.004593 |                           0.004891 |                    0.017578 |                   -0.025462 |

## Score diagnostics

| period   | score_type   |   mean_daily_rankic |   median_daily_rankic |   date_count |
|:---------|:-------------|--------------------:|----------------------:|-------------:|
| stitched | Factor Mean  |            0.050773 |              0.061099 |         1130 |
| stitched | MLP          |            0.028184 |              0.027283 |         1130 |

| score_type   |   d10_minus_d1 |   d10_minus_d5 |   monotonicity_spearman |   non_decreasing_adjacent_fraction |
|:-------------|---------------:|---------------:|------------------------:|-----------------------------------:|
| Factor Mean  |       0.000758 |       0.000171 |                0.781818 |                           0.555556 |
| MLP          |       0.000562 |      -0.000099 |                0.600000 |                           0.555556 |

| period   |   date_count |   spearman_score_correlation_mean |   spearman_score_correlation_median |   spearman_score_correlation_p10 |   spearman_score_correlation_p90 |   pearson_zscore_correlation_mean |   pearson_zscore_correlation_median |   pearson_zscore_correlation_p10 |   pearson_zscore_correlation_p90 |
|:---------|-------------:|----------------------------------:|------------------------------------:|---------------------------------:|---------------------------------:|----------------------------------:|------------------------------------:|---------------------------------:|---------------------------------:|
| stitched |         1130 |                          0.566772 |                            0.589314 |                         0.332617 |                         0.770583 |                          0.691243 |                            0.739000 |                         0.463011 |                         0.855806 |

| period   |   mean_top20_overlap |   median_top20_overlap |   mean_intersection_count |   mean_fm_only_count |   mean_mlp_only_count |   date_count |
|:---------|---------------------:|-----------------------:|--------------------------:|---------------------:|----------------------:|-------------:|
| stitched |             0.196451 |               0.190476 |                 31.985841 |            67.754867 |             67.754867 |         1130 |

## Annual stability

| period   | strategy    |   rankic |   annual_return_7p5bps |   sharpe_7p5bps |   turnover_mean |   effective_n |   annual_excess_return |   information_ratio |
|:---------|:------------|---------:|-----------------------:|----------------:|----------------:|--------------:|-----------------------:|--------------------:|
| stitched | FM-Top20    | 0.050773 |               0.023158 |        0.219837 |        0.089766 |     99.740708 |               0.010750 |            0.049756 |
| 2022     | FM-Top20    | 0.058633 |              -0.112859 |       -0.635300 |        0.081243 |     99.710744 |               0.071277 |            1.171262 |
| 2023     | FM-Top20    | 0.059740 |              -0.032507 |       -0.274718 |        0.085811 |     99.971074 |               0.060227 |            1.266323 |
| 2024     | FM-Top20    | 0.046686 |               0.047724 |        0.308510 |        0.103288 |     99.987603 |               0.019609 |            0.171214 |
| 2025     | FM-Top20    | 0.040863 |               0.245169 |        1.854753 |        0.091340 |    100.000000 |              -0.177056 |           -1.424764 |
| 2026 YTD | FM-Top20    | 0.046579 |              -0.010614 |        0.001366 |        0.085822 |     98.677019 |               0.023364 |            0.035689 |
| stitched | FM-Softmax  | 0.050773 |               0.037923 |        0.302396 |        0.042967 |    319.095338 |               0.025516 |            0.248877 |
| 2022     | FM-Softmax  | 0.058633 |              -0.095258 |       -0.489160 |        0.041583 |    313.086842 |               0.088878 |            2.243887 |
| 2023     | FM-Softmax  | 0.059740 |              -0.034811 |       -0.299631 |        0.041399 |    322.636530 |               0.057923 |            1.737404 |
| 2024     | FM-Softmax  | 0.046686 |               0.047313 |        0.306103 |        0.044309 |    333.245361 |               0.019197 |            0.287473 |
| 2025     | FM-Softmax  | 0.040863 |               0.307368 |        2.125276 |        0.040651 |    322.245091 |              -0.114857 |           -1.303217 |
| 2026 YTD | FM-Softmax  | 0.046579 |              -0.008993 |        0.012773 |        0.048887 |    296.780999 |               0.024986 |            0.055752 |
| stitched | MLP-Top20   | 0.028184 |              -0.004105 |        0.074874 |        0.144474 |     99.740708 |              -0.016512 |           -0.216829 |
| 2022     | MLP-Top20   | 0.034387 |              -0.107858 |       -0.509322 |        0.093251 |     99.710744 |               0.076278 |            1.684049 |
| 2023     | MLP-Top20   | 0.006657 |              -0.116414 |       -0.890230 |        0.135930 |     99.971074 |              -0.023679 |           -0.388058 |
| 2024     | MLP-Top20   | 0.024520 |              -0.015580 |        0.084800 |        0.151474 |     99.987603 |              -0.043696 |           -0.524127 |
| 2025     | MLP-Top20   | 0.041251 |               0.263433 |        1.816963 |        0.149320 |    100.000000 |              -0.158792 |           -1.477175 |
| 2026 YTD | MLP-Top20   | 0.037001 |              -0.000611 |        0.090481 |        0.216475 |     98.677019 |               0.033368 |            0.220100 |
| stitched | MLP-Softmax | 0.028184 |               0.015552 |        0.175011 |        0.072213 |    299.924891 |               0.003144 |            0.023452 |
| 2022     | MLP-Softmax | 0.034387 |              -0.098161 |       -0.480530 |        0.041916 |    315.869773 |               0.085975 |            3.287604 |
| 2023     | MLP-Softmax | 0.006657 |              -0.076524 |       -0.634475 |        0.079805 |    257.381537 |               0.016210 |            0.487612 |
| 2024     | MLP-Softmax | 0.024520 |               0.013298 |        0.180609 |        0.063835 |    336.096100 |              -0.014817 |           -0.274111 |
| 2025     | MLP-Softmax | 0.041251 |               0.317802 |        2.079059 |        0.063975 |    335.664969 |              -0.104423 |           -1.320691 |
| 2026 YTD | MLP-Softmax | 0.037001 |              -0.051718 |       -0.200910 |        0.131371 |    231.592959 |              -0.017739 |           -0.276026 |

All annual values are slices of the one continuous stitched path; they are not averages of isolated fold-level metrics.
