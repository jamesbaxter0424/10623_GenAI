# Full-city dataset QA

## Headline numbers per borough

|                         |        MN |         BX |         BK |         QN |         SI |         NYC |
|:------------------------|----------:|-----------:|-----------:|-----------:|-----------:|------------:|
| footprints_raw          | 45172.000 | 104264.000 | 330097.000 | 460942.000 | 142458.000 | 1082933.000 |
| buildings_kept          | 44916.000 | 103917.000 | 329637.000 | 460250.000 | 142224.000 | 1080944.000 |
| link_rate               |     1.000 |      1.000 |      1.000 |      1.000 |      1.000 |     nan     |
| floor_h_m               |     3.715 |      3.978 |      3.935 |      3.880 |      4.066 |     nan     |
| blocks                  |  2942.000 |   4211.000 |   8398.000 |  12767.000 |   5019.000 |   33337.000 |
| valid_blocks            |  2439.000 |   3819.000 |   7799.000 |  12083.000 |   4726.000 |   30866.000 |
| valid_%                 |    82.903 |     90.691 |     92.867 |     94.642 |     94.162 |      92.588 |
| corr_FAR_valid          |     0.967 |      0.964 |      0.960 |      0.963 |      0.917 |       0.978 |
| median_far_fp/far_pluto |     1.008 |      0.996 |      1.018 |      1.226 |      1.258 |     nan     |
| corr_cov_vec_ras        |     0.998 |      0.998 |      0.996 |      0.997 |      0.997 |       0.998 |


## Blocks removed by each check (core blocks)

| step              |   BK |   BX |   MN |   QN |   SI |
|:------------------|-----:|-----:|-----:|-----:|-----:|
| bldg_in_piece     |    0 |    0 |    0 |    0 |    0 |
| core_blocks       |    0 |    0 |    0 |    0 |    0 |
| flag_construction |  115 |   45 |   32 |   44 |   14 |
| flag_coverage_bad |    0 |    0 |    0 |    0 |    0 |
| flag_far_mismatch |  182 |  103 |  279 |  135 |    1 |
| flag_incomplete   |    0 |    0 |    0 |    0 |    0 |
| flag_no_pluto     |    3 |    7 |    3 |    3 |    7 |
| flag_thin         |    8 |    6 |    5 |   22 |   25 |
| flag_too_big      |   44 |   47 |   26 |   93 |   89 |
| flag_too_small    |  149 |  112 |   94 |  247 |   97 |
| flag_water        |   98 |   72 |   64 |  140 |   60 |


## Split by community district

seed 0; val CDs: [108, 206, 307, 308, 403]; test CDs: [103, 110, 210, 211, 305, 311, 312, 405, 407, 502]

| split   |   valid_blocks |   samples |   samples_clean_ctx |
|:--------|---------------:|----------:|--------------------:|
| test    |           6902 |     19380 |               19380 |
| train   |          22519 |     61621 |               55813 |
| val     |           1445 |      3980 |                3825 |


Samples by group size:

| k   |   test |   train |   val |   All |
|:----|-------:|--------:|------:|------:|
| 1   |   6813 |   22175 |  1436 | 30424 |
| 2   |   5278 |   16898 |  1114 | 23290 |
| 3   |   4329 |   13568 |   873 | 18770 |
| 4   |   2960 |    8980 |   557 | 12497 |
| All |  19380 |   61621 |  3980 | 84981 |


Valid blocks used as a target: 30424 / 30866; appearances per block median 6, max 21

## Median block targets per split

| split   |   coverage |   far_fp |   median_floors |   res_share |   area_ha |
|:--------|-----------:|---------:|----------------:|------------:|----------:|
| test    |      0.385 |    0.775 |           2.000 |       0.861 |     0.999 |
| train   |      0.393 |    0.809 |           2.000 |       0.874 |     1.037 |
| val     |      0.543 |    1.567 |           3.000 |       0.811 |     1.155 |


Height class share per split (0: 1-2, 1: 3-5, 2: 6-12, 3: 13+ floors):

| split   |    -1 |     0 |     1 |     2 |     3 |
|:--------|------:|------:|------:|------:|------:|
| test    | 0.034 | 0.754 | 0.170 | 0.038 | 0.004 |
| train   | 0.028 | 0.644 | 0.232 | 0.077 | 0.020 |
| val     | 0.020 | 0.398 | 0.390 | 0.134 | 0.058 |


## Block graph

edges 77760, mean degree 4.67, isolated blocks 72, touching (no street between) 3915; street width median 21.9 m, facing length median 112 m

## Class shares on land

| class    |   share |
|:---------|--------:|
| street   |  0.2457 |
| open     |  0.5435 |
| res_1-2  |  0.0795 |
| res_3-5  |  0.0423 |
| res_6-12 |  0.0171 |
| res_13+  |  0.0055 |
| non_1-2  |  0.0362 |
| non_3-5  |  0.0186 |
| non_6-12 |  0.0075 |
| non_13+  |  0.0042 |

