# Pilot QA summary

## Funnel (row counts per cleaning step)

| pilot      | stage   | step                |   n_in |   n_out |   dropped | note                                                                                     |   drop_% |
|:-----------|:--------|:--------------------|-------:|--------:|----------:|:-----------------------------------------------------------------------------------------|---------:|
| park_slope | 1_lots  | pluto_valid_bbl     |  14142 |   14142 |         0 | nan                                                                                      |      0   |
| park_slope | 1_lots  | pluto_dedupe_bbl    |  14142 |   14142 |         0 | nan                                                                                      |      0   |
| park_slope | 1_lots  | taxlot_valid_geom   |  12807 |   12807 |         0 | nan                                                                                      |      0   |
| park_slope | 1_lots  | taxlot_drop_air_sub |  12807 |   12807 |         0 | 7 flagged, 7 kept (no base lot below)                                                    |      0   |
| park_slope | 1_lots  | taxlot_dedupe_bbl   |  12807 |   12807 |         0 | dissolve duplicate BBL                                                                   |      0   |
| park_slope | 1_lots  | taxlot_pluto_join   |  12807 |   12755 |        52 | {'bbl': 12006, 'point_in_lot': 749, 'none': 52}                                          |      0.4 |
| midtown    | 1_lots  | pluto_valid_bbl     |   4637 |    4637 |         0 | nan                                                                                      |      0   |
| midtown    | 1_lots  | pluto_dedupe_bbl    |   4637 |    4637 |         0 | nan                                                                                      |      0   |
| midtown    | 1_lots  | taxlot_valid_geom   |   3996 |    3996 |         0 | nan                                                                                      |      0   |
| midtown    | 1_lots  | taxlot_drop_air_sub |   3996 |    3996 |         0 | 69 flagged, 69 kept (no base lot below)                                                  |      0   |
| midtown    | 1_lots  | taxlot_dedupe_bbl   |   3996 |    3996 |         0 | dissolve duplicate BBL                                                                   |      0   |
| midtown    | 1_lots  | taxlot_pluto_join   |   3996 |    3993 |         3 | {'bbl': 3685, 'point_in_lot': 308, 'none': 3}                                            |      0.1 |
| lic        | 1_lots  | pluto_valid_bbl     |   4450 |    4450 |         0 | nan                                                                                      |      0   |
| lic        | 1_lots  | pluto_dedupe_bbl    |   4450 |    4450 |         0 | nan                                                                                      |      0   |
| lic        | 1_lots  | taxlot_valid_geom   |   3910 |    3910 |         0 | nan                                                                                      |      0   |
| lic        | 1_lots  | taxlot_drop_air_sub |   3910 |    3910 |         0 | 8 flagged, 8 kept (no base lot below)                                                    |      0   |
| lic        | 1_lots  | taxlot_dedupe_bbl   |   3910 |    3910 |         0 | dissolve duplicate BBL                                                                   |      0   |
| lic        | 1_lots  | taxlot_pluto_join   |   3910 |    3908 |         2 | {'bbl': 3773, 'point_in_lot': 135, 'none': 2}                                            |      0.1 |
| park_slope | 2_bldg  | feature_code        |  13003 |   12977 |        26 | all codes {2100: 12722, 5110: 255, 5100: 16, 1004: 6, 1000: 2, 2110: 2}                  |      0.2 |
| park_slope | 2_bldg  | valid_geom          |  12977 |   12977 |         0 | nan                                                                                      |      0   |
| park_slope | 2_bldg  | area>=10.0m2        |  12977 |   12977 |         0 | nan                                                                                      |      0   |
| park_slope | 2_bldg  | link_pluto          |  12977 |   12974 |         3 | {'mappluto': 12974, 'none': 3}                                                           |      0   |
| park_slope | 2_bldg  | inside_a_lot        |  12977 |   12971 |         6 | footprint point not in any lot -> drop                                                   |      0   |
| park_slope | 2_bldg  | has_floors          |  12971 |   12971 |         0 | nan                                                                                      |      0   |
| midtown    | 2_bldg  | feature_code        |   4178 |    4153 |        25 | all codes {2100: 4140, 5110: 13, 1000: 10, 5100: 8, 1003: 3, 1004: 2, 1005: 1, 2110: 1}  |      0.6 |
| midtown    | 2_bldg  | valid_geom          |   4153 |    4153 |         0 | nan                                                                                      |      0   |
| midtown    | 2_bldg  | area>=10.0m2        |   4153 |    4152 |         1 | nan                                                                                      |      0   |
| midtown    | 2_bldg  | link_pluto          |   4152 |    4151 |         1 | {'mappluto': 4151, 'none': 1}                                                            |      0   |
| midtown    | 2_bldg  | inside_a_lot        |   4152 |    4151 |         1 | footprint point not in any lot -> drop                                                   |      0   |
| midtown    | 2_bldg  | has_floors          |   4151 |    4150 |         1 | nan                                                                                      |      0   |
| lic        | 2_bldg  | feature_code        |   4268 |    4234 |        34 | all codes {2100: 3970, 5110: 264, 5100: 14, 1004: 7, 1000: 5, 1001: 5, 1003: 2, 2110: 1} |      0.8 |
| lic        | 2_bldg  | valid_geom          |   4234 |    4234 |         0 | nan                                                                                      |      0   |
| lic        | 2_bldg  | area>=10.0m2        |   4234 |    4234 |         0 | nan                                                                                      |      0   |
| lic        | 2_bldg  | link_pluto          |   4234 |    4232 |         2 | {'mappluto': 4232, 'none': 2}                                                            |      0   |
| lic        | 2_bldg  | inside_a_lot        |   4234 |    4231 |         3 | footprint point not in any lot -> drop                                                   |      0.1 |
| lic        | 2_bldg  | has_floors          |   4231 |    4229 |         2 | nan                                                                                      |      0   |
| park_slope | 3_block | bldg_in_piece       |  12971 |   12971 |         0 | nan                                                                                      |      0   |
| park_slope | 3_block | core_blocks         |    346 |     184 |       162 | blocks whose point is in core bbox                                                       |     46.8 |
| park_slope | 3_block | flag_incomplete     |    184 |     184 |         0 | nan                                                                                      |      0   |
| park_slope | 3_block | flag_no_pluto       |    184 |     184 |         0 | nan                                                                                      |      0   |
| park_slope | 3_block | flag_too_big        |    184 |     184 |         0 | nan                                                                                      |      0   |
| park_slope | 3_block | flag_too_small      |    184 |     183 |         1 | nan                                                                                      |      0.5 |
| park_slope | 3_block | flag_thin           |    183 |     183 |         0 | nan                                                                                      |      0   |
| park_slope | 3_block | flag_water          |    183 |     183 |         0 | nan                                                                                      |      0   |
| park_slope | 3_block | flag_coverage_bad   |    183 |     183 |         0 | nan                                                                                      |      0   |
| park_slope | 3_block | flag_construction   |    183 |     175 |         8 | nan                                                                                      |      4.4 |
| park_slope | 3_block | flag_far_mismatch   |    175 |     168 |         7 | nan                                                                                      |      4   |
| midtown    | 3_block | bldg_in_piece       |   4150 |    4150 |         0 | nan                                                                                      |      0   |
| midtown    | 3_block | core_blocks         |    326 |     150 |       176 | blocks whose point is in core bbox                                                       |     54   |
| midtown    | 3_block | flag_incomplete     |    150 |     150 |         0 | nan                                                                                      |      0   |
| midtown    | 3_block | flag_no_pluto       |    150 |     150 |         0 | nan                                                                                      |      0   |
| midtown    | 3_block | flag_too_big        |    150 |     150 |         0 | nan                                                                                      |      0   |
| midtown    | 3_block | flag_too_small      |    150 |     145 |         5 | nan                                                                                      |      3.3 |
| midtown    | 3_block | flag_thin           |    145 |     145 |         0 | nan                                                                                      |      0   |
| midtown    | 3_block | flag_water          |    145 |     145 |         0 | nan                                                                                      |      0   |
| midtown    | 3_block | flag_coverage_bad   |    145 |     145 |         0 | nan                                                                                      |      0   |
| midtown    | 3_block | flag_construction   |    145 |     142 |         3 | nan                                                                                      |      2.1 |
| midtown    | 3_block | flag_far_mismatch   |    142 |     118 |        24 | nan                                                                                      |     16.9 |
| lic        | 3_block | bldg_in_piece       |   4229 |    4229 |         0 | nan                                                                                      |      0   |
| lic        | 3_block | core_blocks         |    365 |     213 |       152 | blocks whose point is in core bbox                                                       |     41.6 |
| lic        | 3_block | flag_incomplete     |    213 |     213 |         0 | nan                                                                                      |      0   |
| lic        | 3_block | flag_no_pluto       |    213 |     213 |         0 | nan                                                                                      |      0   |
| lic        | 3_block | flag_too_big        |    213 |     211 |         2 | nan                                                                                      |      0.9 |
| lic        | 3_block | flag_too_small      |    211 |     204 |         7 | nan                                                                                      |      3.3 |
| lic        | 3_block | flag_thin           |    204 |     204 |         0 | nan                                                                                      |      0   |
| lic        | 3_block | flag_water          |    204 |     204 |         0 | nan                                                                                      |      0   |
| lic        | 3_block | flag_coverage_bad   |    204 |     204 |         0 | nan                                                                                      |      0   |
| lic        | 3_block | flag_construction   |    204 |     193 |        11 | nan                                                                                      |      5.4 |
| lic        | 3_block | flag_far_mismatch   |    193 |     165 |        28 | nan                                                                                      |     14.5 |


**Steps dropping >10% (check):**

| pilot   | stage   | step              |   drop_% |
|:--------|:--------|:------------------|---------:|
| midtown | 3_block | flag_far_mismatch |     16.9 |
| lic     | 3_block | flag_far_mismatch |     14.5 |


## Headline checks

|                                  | park_slope            | midtown               | lic                   |
|:---------------------------------|:----------------------|:----------------------|:----------------------|
| buildings                        | 12971                 | 4150                  | 4229                  |
| link_rate                        | 1.0                   | 1.0                   | 1.0                   |
| link_mappluto/base/spatial       | 1.00/0.00/0.00        | 1.00/0.00/0.00        | 1.00/0.00/0.00        |
| height_missing                   | 0.0004625703492406137 | 0.0012048192771084338 | 0.0014187751241428233 |
| floor_h_m                        | 3.99                  | 3.72                  | 3.97                  |
| floor_disagree(single-bldg lots) | 0.0067388893627910946 | 0.16206896551724137   | 0.029116790682626983  |
| valid_blocks                     | 168                   | 118                   | 165                   |
| corr_FAR_before_mismatch_drop    | 0.7651465667442016    | 0.8681641967156118    | 0.8705254158810183    |
| corr_FAR_valid                   | 0.9446231819455401    | 0.9406279324802604    | 0.9761929288007678    |
| median_lotarea/poly              | 0.9684032577994669    | 0.960944801188398     | 0.9763141734309695    |
| corr_cov_vec_vs_ras              | 0.9968266354051812    | 0.9966994108482577    | 0.9995807287570809    |
| mean|cov_vec-ras|                | 0.0100213882296       | 0.00882084721515314   | 0.006963653362510839  |
| bldg_vanished                    | 0.0003213883978788745 | 0.000589622641509413  | 0.0                   |
| tiles                            | 12                    | 9                     | 12                    |
| tiles_water>10%                  | 0                     | 0                     | 1                     |


Targets: link_rate ≥ 0.90; corr FAR ≈ 0.88 (Park Slope); lotarea/poly ≈ 1; |cov_vec − ras| small; floor height 2.5–5 m.

## Blocks to inspect by hand (largest FAR disagreement)

### park_slope

| block_id   |   area_ha |   coverage |   far_pluto |   far_fp |   median_floors |   res_share |   n_bldg |
|:-----------|----------:|-----------:|------------:|---------:|----------------:|------------:|---------:|
| 3-01128-a  |      1.73 |       0.65 |        4.66 |     6.26 |            9.00 |        0.80 |    29.00 |
| 3-00452-a  |      0.74 |       0.49 |        3.36 |     4.81 |           12.00 |        0.95 |     2.00 |
| 3-01172-a  |      0.99 |       0.64 |        3.92 |     5.21 |            6.00 |        0.82 |    16.00 |
| 3-00440-a  |      1.00 |       0.84 |        3.77 |     4.71 |            9.00 |        0.82 |    18.00 |
| 3-01084-a  |      1.37 |       0.77 |        5.20 |     4.38 |            7.00 |        0.06 |    13.00 |
| 3-01122-a  |      1.45 |       0.53 |        2.74 |     3.53 |            4.00 |        0.71 |    41.00 |
| 3-00968-a  |      2.85 |       0.34 |        0.00 |     0.79 |            3.00 |        0.00 |     6.00 |
| 3-00406-a  |      1.05 |       0.50 |        1.43 |     2.20 |            2.00 |        0.52 |    10.00 |


### midtown

| block_id   |   area_ha |   coverage |   far_pluto |   far_fp |   median_floors |   res_share |   n_bldg |
|:-----------|----------:|-----------:|------------:|---------:|----------------:|------------:|---------:|
| 1-01277-b  |      0.43 |       0.94 |       38.10 |    56.20 |           60.00 |        0.00 |     1.00 |
| 1-00869-b  |      0.77 |       0.86 |       18.39 |    27.11 |           41.00 |        0.00 |     9.00 |
| 1-01277-a  |      0.82 |       0.91 |       18.34 |    26.86 |           28.00 |        0.00 |     7.00 |
| 1-01275-b  |      0.80 |       0.86 |       19.41 |    27.73 |           36.00 |        0.00 |     5.00 |
| 1-01268-a  |      1.76 |       0.75 |       18.26 |    26.51 |           40.00 |        0.05 |     6.00 |
| 1-01002-a  |      1.49 |       0.59 |       17.51 |    25.74 |           54.00 |        0.00 |     2.00 |
| 1-01282-a  |      0.80 |       0.91 |       19.05 |    26.89 |           32.00 |        0.00 |     4.00 |
| 1-01000-a  |      1.55 |       0.73 |       20.37 |    27.97 |           45.00 |        0.00 |     8.00 |


### lic

| block_id   |   area_ha |   coverage |   far_pluto |   far_fp |   median_floors |   res_share |   n_bldg |
|:-----------|----------:|-----------:|------------:|---------:|----------------:|------------:|---------:|
| 4-00425-a  |      0.41 |       0.95 |       11.84 |    17.35 |            4.00 |        0.76 |     2.00 |
| 4-00418-a  |      0.40 |       0.83 |       11.22 |    15.87 |           16.00 |        0.54 |     7.00 |
| 4-00413-a  |      0.94 |       0.67 |        7.34 |    10.45 |           17.00 |        0.87 |     9.00 |
| 4-00267-a  |      1.06 |       0.41 |        7.43 |     9.64 |           26.00 |        0.96 |     7.00 |
| 4-00417-a  |      0.83 |       0.84 |        4.91 |     7.08 |            9.00 |        0.44 |    14.00 |
| 4-00302-a  |      0.13 |       0.64 |        5.01 |     6.67 |           12.00 |        0.00 |     2.00 |
| 4-00434-a  |      0.25 |       0.75 |        5.98 |     7.32 |            2.00 |        0.89 |     4.00 |
| 4-00082-a  |      0.73 |       0.78 |        4.94 |     6.25 |            8.00 |        0.82 |     1.00 |

