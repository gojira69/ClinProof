# Section 2: Results (Full Granularity)

## BIOASQ

### Overall Metrics

| ID | Tag | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) |
|---|---|---|---|---|---|
| A3 | A3_biomistral_bioasq | 0.819 | 0.410 | 0.500 | 0.450 |
| A5 | A5_mistral7b_bioasq | 0.822 | 0.676 | 0.595 | 0.612 |
| A6 | A6_qwen14b_nokgbm25_bioasq | 0.795 | 0.664 | 0.680 | 0.671 |
| B1 | B1_dense_only_bioasq | 0.873 | 0.794 | 0.754 | 0.771 |
| B2 | B2_dense_kg_bioasq | 0.843 | 0.736 | 0.749 | 0.742 |
| B3 | B3_dense_bm25_bioasq | 0.904 | 0.849 | 0.811 | 0.828 |
| B4 | B4_full_pipeline_bioasq | 0.880 | 0.799 | 0.784 | 0.791 |
| BEST1 | BEST1_homo_dense_bm25_bioasq | 0.880 | 0.808 | 0.758 | 0.779 |
| BEST2 | BEST2_hetero_dense_bm25_bioasq | 0.843 | 0.762 | 0.619 | 0.646 |
| D1 | D1_qwen14b_1vote_bioasq | 0.771 | 0.639 | 0.665 | 0.649 |
| D2 | D2_qwen14b_3vote_bioasq | 0.801 | 0.671 | 0.684 | 0.677 |
| D4 | D4_medensemble_3_bioasq | 0.819 | 0.410 | 0.500 | 0.450 |
| D5 | D5_hybridensemble_3_bioasq | 0.837 | 0.727 | 0.641 | 0.665 |
| Z1 | Z1_zeroshot_baseline_bioasq | 0.861 | 0.767 | 0.825 | 0.790 |

### A3 - A3_biomistral_bioasq

#### Class-wise Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| yes | 0.819 | 1.000 | 0.901 | 136 |
| no | 0.000 | 0.000 | 0.000 | 30 |

#### Confusion Matrix (True \ Pred)

| True \ Pred | yes | no |
|---|---|---|
| **yes** | 136 | 0 |
| **no** | 30 | 0 |

#### Per-class Error Breakdown

- **yes**: 0 errors out of 136 (0.0%)
- **no**: 30 errors out of 30 (100.0%)
  - Misclassified as **yes**: 30

---

### A5 - A5_mistral7b_bioasq

#### Class-wise Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| yes | 0.852 | 0.948 | 0.898 | 134 |
| no | 0.500 | 0.241 | 0.326 | 29 |

#### Confusion Matrix (True \ Pred)

| True \ Pred | yes | no |
|---|---|---|
| **yes** | 127 | 7 |
| **no** | 22 | 7 |

#### Per-class Error Breakdown

- **yes**: 7 errors out of 134 (5.2%)
  - Misclassified as **no**: 7
- **no**: 22 errors out of 29 (75.9%)
  - Misclassified as **yes**: 22

---

### A6 - A6_qwen14b_nokgbm25_bioasq

#### Class-wise Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| yes | 0.886 | 0.860 | 0.873 | 136 |
| no | 0.441 | 0.500 | 0.469 | 30 |

#### Confusion Matrix (True \ Pred)

| True \ Pred | yes | no |
|---|---|---|
| **yes** | 117 | 19 |
| **no** | 15 | 15 |

#### Per-class Error Breakdown

- **yes**: 19 errors out of 136 (14.0%)
  - Misclassified as **no**: 19
- **no**: 15 errors out of 30 (50.0%)
  - Misclassified as **yes**: 15

---

### B1 - B1_dense_only_bioasq

#### Class-wise Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| yes | 0.908 | 0.941 | 0.924 | 136 |
| no | 0.680 | 0.567 | 0.618 | 30 |

#### Confusion Matrix (True \ Pred)

| True \ Pred | yes | no |
|---|---|---|
| **yes** | 128 | 8 |
| **no** | 13 | 17 |

#### Per-class Error Breakdown

- **yes**: 8 errors out of 136 (5.9%)
  - Misclassified as **no**: 8
- **no**: 13 errors out of 30 (43.3%)
  - Misclassified as **yes**: 13

---

### B2 - B2_dense_kg_bioasq

#### Class-wise Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| yes | 0.910 | 0.897 | 0.904 | 136 |
| no | 0.562 | 0.600 | 0.581 | 30 |

#### Confusion Matrix (True \ Pred)

| True \ Pred | yes | no |
|---|---|---|
| **yes** | 122 | 14 |
| **no** | 12 | 18 |

#### Per-class Error Breakdown

- **yes**: 14 errors out of 136 (10.3%)
  - Misclassified as **no**: 14
- **no**: 12 errors out of 30 (40.0%)
  - Misclassified as **yes**: 12

---

### B3 - B3_dense_bm25_bioasq

#### Class-wise Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| yes | 0.929 | 0.956 | 0.942 | 136 |
| no | 0.769 | 0.667 | 0.714 | 30 |

#### Confusion Matrix (True \ Pred)

| True \ Pred | yes | no |
|---|---|---|
| **yes** | 130 | 6 |
| **no** | 10 | 20 |

#### Per-class Error Breakdown

- **yes**: 6 errors out of 136 (4.4%)
  - Misclassified as **no**: 6
- **no**: 10 errors out of 30 (33.3%)
  - Misclassified as **yes**: 10

---

### B4 - B4_full_pipeline_bioasq

#### Class-wise Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| yes | 0.920 | 0.934 | 0.927 | 136 |
| no | 0.679 | 0.633 | 0.655 | 30 |

#### Confusion Matrix (True \ Pred)

| True \ Pred | yes | no |
|---|---|---|
| **yes** | 127 | 9 |
| **no** | 11 | 19 |

#### Per-class Error Breakdown

- **yes**: 9 errors out of 136 (6.6%)
  - Misclassified as **no**: 9
- **no**: 11 errors out of 30 (36.7%)
  - Misclassified as **yes**: 11

---

### BEST1 - BEST1_homo_dense_bm25_bioasq

#### Class-wise Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| yes | 0.908 | 0.949 | 0.928 | 136 |
| no | 0.708 | 0.567 | 0.630 | 30 |

#### Confusion Matrix (True \ Pred)

| True \ Pred | yes | no |
|---|---|---|
| **yes** | 129 | 7 |
| **no** | 13 | 17 |

#### Per-class Error Breakdown

- **yes**: 7 errors out of 136 (5.1%)
  - Misclassified as **no**: 7
- **no**: 13 errors out of 30 (43.3%)
  - Misclassified as **yes**: 13

---

### BEST2 - BEST2_hetero_dense_bm25_bioasq

#### Class-wise Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| yes | 0.857 | 0.971 | 0.910 | 136 |
| no | 0.667 | 0.267 | 0.381 | 30 |

#### Confusion Matrix (True \ Pred)

| True \ Pred | yes | no |
|---|---|---|
| **yes** | 132 | 4 |
| **no** | 22 | 8 |

#### Per-class Error Breakdown

- **yes**: 4 errors out of 136 (2.9%)
  - Misclassified as **no**: 4
- **no**: 22 errors out of 30 (73.3%)
  - Misclassified as **yes**: 22

---

### D1 - D1_qwen14b_1vote_bioasq

#### Class-wise Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| yes | 0.883 | 0.831 | 0.856 | 136 |
| no | 0.395 | 0.500 | 0.441 | 30 |

#### Confusion Matrix (True \ Pred)

| True \ Pred | yes | no |
|---|---|---|
| **yes** | 113 | 23 |
| **no** | 15 | 15 |

#### Per-class Error Breakdown

- **yes**: 23 errors out of 136 (16.9%)
  - Misclassified as **no**: 23
- **no**: 15 errors out of 30 (50.0%)
  - Misclassified as **yes**: 15

---

### D2 - D2_qwen14b_3vote_bioasq

#### Class-wise Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| yes | 0.887 | 0.868 | 0.877 | 136 |
| no | 0.455 | 0.500 | 0.476 | 30 |

#### Confusion Matrix (True \ Pred)

| True \ Pred | yes | no |
|---|---|---|
| **yes** | 118 | 18 |
| **no** | 15 | 15 |

#### Per-class Error Breakdown

- **yes**: 18 errors out of 136 (13.2%)
  - Misclassified as **no**: 18
- **no**: 15 errors out of 30 (50.0%)
  - Misclassified as **yes**: 15

---

### D4 - D4_medensemble_3_bioasq

#### Class-wise Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| yes | 0.819 | 1.000 | 0.901 | 136 |
| no | 0.000 | 0.000 | 0.000 | 30 |

#### Confusion Matrix (True \ Pred)

| True \ Pred | yes | no |
|---|---|---|
| **yes** | 136 | 0 |
| **no** | 30 | 0 |

#### Per-class Error Breakdown

- **yes**: 0 errors out of 136 (0.0%)
- **no**: 30 errors out of 30 (100.0%)
  - Misclassified as **yes**: 30

---

### D5 - D5_hybridensemble_3_bioasq

#### Class-wise Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| yes | 0.866 | 0.949 | 0.905 | 136 |
| no | 0.588 | 0.333 | 0.426 | 30 |

#### Confusion Matrix (True \ Pred)

| True \ Pred | yes | no |
|---|---|---|
| **yes** | 129 | 7 |
| **no** | 20 | 10 |

#### Per-class Error Breakdown

- **yes**: 7 errors out of 136 (5.1%)
  - Misclassified as **no**: 7
- **no**: 20 errors out of 30 (66.7%)
  - Misclassified as **yes**: 20

---

### Z1 - Z1_zeroshot_baseline_bioasq

#### Class-wise Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| yes | 0.945 | 0.882 | 0.913 | 136 |
| no | 0.590 | 0.767 | 0.667 | 30 |

#### Confusion Matrix (True \ Pred)

| True \ Pred | yes | no |
|---|---|---|
| **yes** | 120 | 16 |
| **no** | 7 | 23 |

#### Per-class Error Breakdown

- **yes**: 16 errors out of 136 (11.8%)
  - Misclassified as **no**: 16
- **no**: 7 errors out of 30 (23.3%)
  - Misclassified as **yes**: 7

---

## HEALTHFC

### Overall Metrics

| ID | Tag | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) |
|---|---|---|---|---|---|
| B1 | B1_dense_only_healthfc_test | 0.307 | 0.431 | 0.324 | 0.312 |
| B2 | B2_dense_kg_healthfc_test | 0.320 | 0.459 | 0.357 | 0.340 |
| B3 | B3_dense_bm25_healthfc_test | 0.320 | 0.438 | 0.355 | 0.315 |
| B4 | B4_full_pipeline_healthfc_test | 0.320 | 0.465 | 0.340 | 0.335 |
| BEST1 | BEST1_homo_dense_bm25_healthfc_test | 0.333 | 0.466 | 0.364 | 0.342 |
| BEST2 | BEST2_hetero_dense_bm25_healthfc_test | 0.360 | 0.558 | 0.486 | 0.358 |
| E1 | E1_qwen14b_3vote_with_decomp_healthfc_test | 0.307 | 0.459 | 0.341 | 0.327 |
| E2 | E2_qwen14b_3vote_no_decomp_healthfc_test | 0.453 | 0.611 | 0.363 | 0.330 |
| LIVE1 | LIVE1_base_homo_healthfc_test | 0.387 | 0.531 | 0.405 | 0.399 |
| LIVE2 | LIVE2_live_only_rag_healthfc_test | 0.320 | 0.552 | 0.403 | 0.341 |
| Z1 | Z1_zeroshot_baseline_healthfc_test | 0.413 | 0.523 | 0.436 | 0.400 |
| sanity | sanity_nocompression_healthfc_test | 0.433 | 0.598 | 0.489 | 0.428 |

### B1 - B1_dense_only_healthfc_test

#### Class-wise Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| true | 0.500 | 0.250 | 0.333 | 20 |
| false | 0.667 | 0.293 | 0.407 | 41 |
| mixture | 0.128 | 0.429 | 0.197 | 14 |

#### Confusion Matrix (True \ Pred)

| True \ Pred | true | false | mixture |
|---|---|---|---|
| **true** | 5 | 1 | 14 |
| **false** | 2 | 12 | 27 |
| **mixture** | 3 | 5 | 6 |

#### Per-class Error Breakdown

- **true**: 15 errors out of 20 (75.0%)
  - Misclassified as **false**: 1
  - Misclassified as **mixture**: 14
- **false**: 29 errors out of 41 (70.7%)
  - Misclassified as **true**: 2
  - Misclassified as **mixture**: 27
- **mixture**: 8 errors out of 14 (57.1%)
  - Misclassified as **true**: 3
  - Misclassified as **false**: 5

---

### B2 - B2_dense_kg_healthfc_test

#### Class-wise Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| true | 0.533 | 0.400 | 0.457 | 20 |
| false | 0.714 | 0.244 | 0.364 | 41 |
| mixture | 0.130 | 0.429 | 0.200 | 14 |

#### Confusion Matrix (True \ Pred)

| True \ Pred | true | false | mixture |
|---|---|---|---|
| **true** | 8 | 0 | 12 |
| **false** | 3 | 10 | 28 |
| **mixture** | 4 | 4 | 6 |

#### Per-class Error Breakdown

- **true**: 12 errors out of 20 (60.0%)
  - Misclassified as **mixture**: 12
- **false**: 31 errors out of 41 (75.6%)
  - Misclassified as **true**: 3
  - Misclassified as **mixture**: 28
- **mixture**: 8 errors out of 14 (57.1%)
  - Misclassified as **true**: 4
  - Misclassified as **false**: 4

---

### B3 - B3_dense_bm25_healthfc_test

#### Class-wise Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| true | 0.444 | 0.200 | 0.276 | 20 |
| false | 0.706 | 0.293 | 0.414 | 41 |
| mixture | 0.163 | 0.571 | 0.254 | 14 |

#### Confusion Matrix (True \ Pred)

| True \ Pred | true | false | mixture |
|---|---|---|---|
| **true** | 4 | 1 | 15 |
| **false** | 3 | 12 | 26 |
| **mixture** | 2 | 4 | 8 |

#### Per-class Error Breakdown

- **true**: 16 errors out of 20 (80.0%)
  - Misclassified as **false**: 1
  - Misclassified as **mixture**: 15
- **false**: 29 errors out of 41 (70.7%)
  - Misclassified as **true**: 3
  - Misclassified as **mixture**: 26
- **mixture**: 6 errors out of 14 (42.9%)
  - Misclassified as **true**: 2
  - Misclassified as **false**: 4

---

### B4 - B4_full_pipeline_healthfc_test

#### Class-wise Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| true | 0.600 | 0.300 | 0.400 | 20 |
| false | 0.667 | 0.293 | 0.407 | 41 |
| mixture | 0.128 | 0.429 | 0.197 | 14 |

#### Confusion Matrix (True \ Pred)

| True \ Pred | true | false | mixture |
|---|---|---|---|
| **true** | 6 | 0 | 14 |
| **false** | 2 | 12 | 27 |
| **mixture** | 2 | 6 | 6 |

#### Per-class Error Breakdown

- **true**: 14 errors out of 20 (70.0%)
  - Misclassified as **mixture**: 14
- **false**: 29 errors out of 41 (70.7%)
  - Misclassified as **true**: 2
  - Misclassified as **mixture**: 27
- **mixture**: 8 errors out of 14 (57.1%)
  - Misclassified as **true**: 2
  - Misclassified as **false**: 6

---

### BEST1 - BEST1_homo_dense_bm25_healthfc_test

#### Class-wise Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| true | 0.500 | 0.300 | 0.375 | 20 |
| false | 0.750 | 0.293 | 0.421 | 41 |
| mixture | 0.149 | 0.500 | 0.230 | 14 |

#### Confusion Matrix (True \ Pred)

| True \ Pred | true | false | mixture |
|---|---|---|---|
| **true** | 6 | 0 | 14 |
| **false** | 3 | 12 | 26 |
| **mixture** | 3 | 4 | 7 |

#### Per-class Error Breakdown

- **true**: 14 errors out of 20 (70.0%)
  - Misclassified as **mixture**: 14
- **false**: 29 errors out of 41 (70.7%)
  - Misclassified as **true**: 3
  - Misclassified as **mixture**: 26
- **mixture**: 7 errors out of 14 (50.0%)
  - Misclassified as **true**: 3
  - Misclassified as **false**: 4

---

### BEST2 - BEST2_hetero_dense_bm25_healthfc_test

#### Class-wise Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| true | 0.423 | 0.550 | 0.478 | 20 |
| false | 1.000 | 0.122 | 0.217 | 41 |
| mixture | 0.250 | 0.786 | 0.379 | 14 |

#### Confusion Matrix (True \ Pred)

| True \ Pred | true | false | mixture |
|---|---|---|---|
| **true** | 11 | 0 | 9 |
| **false** | 12 | 5 | 24 |
| **mixture** | 3 | 0 | 11 |

#### Per-class Error Breakdown

- **true**: 9 errors out of 20 (45.0%)
  - Misclassified as **mixture**: 9
- **false**: 36 errors out of 41 (87.8%)
  - Misclassified as **true**: 12
  - Misclassified as **mixture**: 24
- **mixture**: 3 errors out of 14 (21.4%)
  - Misclassified as **true**: 3

---

### E1 - E1_qwen14b_3vote_with_decomp_healthfc_test

#### Class-wise Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| true | 0.538 | 0.350 | 0.424 | 20 |
| false | 0.714 | 0.244 | 0.364 | 41 |
| mixture | 0.125 | 0.429 | 0.194 | 14 |

#### Confusion Matrix (True \ Pred)

| True \ Pred | true | false | mixture |
|---|---|---|---|
| **true** | 7 | 0 | 13 |
| **false** | 2 | 10 | 29 |
| **mixture** | 4 | 4 | 6 |

#### Per-class Error Breakdown

- **true**: 13 errors out of 20 (65.0%)
  - Misclassified as **mixture**: 13
- **false**: 31 errors out of 41 (75.6%)
  - Misclassified as **true**: 2
  - Misclassified as **mixture**: 29
- **mixture**: 8 errors out of 14 (57.1%)
  - Misclassified as **true**: 4
  - Misclassified as **false**: 4

---

### E2 - E2_qwen14b_3vote_no_decomp_healthfc_test

#### Class-wise Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| true | 1.000 | 0.050 | 0.095 | 20 |
| false | 0.683 | 0.683 | 0.683 | 41 |
| mixture | 0.152 | 0.357 | 0.213 | 14 |

#### Confusion Matrix (True \ Pred)

| True \ Pred | true | false | mixture |
|---|---|---|---|
| **true** | 1 | 4 | 15 |
| **false** | 0 | 28 | 13 |
| **mixture** | 0 | 9 | 5 |

#### Per-class Error Breakdown

- **true**: 19 errors out of 20 (95.0%)
  - Misclassified as **false**: 4
  - Misclassified as **mixture**: 15
- **false**: 13 errors out of 41 (31.7%)
  - Misclassified as **mixture**: 13
- **mixture**: 9 errors out of 14 (64.3%)
  - Misclassified as **false**: 9

---

### LIVE1 - LIVE1_base_homo_healthfc_test

#### Class-wise Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| true | 0.778 | 0.350 | 0.483 | 20 |
| false | 0.652 | 0.366 | 0.469 | 41 |
| mixture | 0.163 | 0.500 | 0.246 | 14 |

#### Confusion Matrix (True \ Pred)

| True \ Pred | true | false | mixture |
|---|---|---|---|
| **true** | 7 | 2 | 11 |
| **false** | 1 | 15 | 25 |
| **mixture** | 1 | 6 | 7 |

#### Per-class Error Breakdown

- **true**: 13 errors out of 20 (65.0%)
  - Misclassified as **false**: 2
  - Misclassified as **mixture**: 11
- **false**: 26 errors out of 41 (63.4%)
  - Misclassified as **true**: 1
  - Misclassified as **mixture**: 25
- **mixture**: 7 errors out of 14 (50.0%)
  - Misclassified as **true**: 1
  - Misclassified as **false**: 6

---

### LIVE2 - LIVE2_live_only_rag_healthfc_test

#### Class-wise Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| true | 0.750 | 0.300 | 0.429 | 20 |
| false | 0.727 | 0.195 | 0.308 | 41 |
| mixture | 0.179 | 0.714 | 0.286 | 14 |

#### Confusion Matrix (True \ Pred)

| True \ Pred | true | false | mixture |
|---|---|---|---|
| **true** | 6 | 0 | 14 |
| **false** | 1 | 8 | 32 |
| **mixture** | 1 | 3 | 10 |

#### Per-class Error Breakdown

- **true**: 14 errors out of 20 (70.0%)
  - Misclassified as **mixture**: 14
- **false**: 33 errors out of 41 (80.5%)
  - Misclassified as **true**: 1
  - Misclassified as **mixture**: 32
- **mixture**: 4 errors out of 14 (28.6%)
  - Misclassified as **true**: 1
  - Misclassified as **false**: 3

---

### Z1 - Z1_zeroshot_baseline_healthfc_test

#### Class-wise Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| true | 0.625 | 0.250 | 0.357 | 20 |
| false | 0.739 | 0.415 | 0.531 | 41 |
| mixture | 0.205 | 0.643 | 0.310 | 14 |

#### Confusion Matrix (True \ Pred)

| True \ Pred | true | false | mixture |
|---|---|---|---|
| **true** | 5 | 3 | 12 |
| **false** | 1 | 17 | 23 |
| **mixture** | 2 | 3 | 9 |

#### Per-class Error Breakdown

- **true**: 15 errors out of 20 (75.0%)
  - Misclassified as **false**: 3
  - Misclassified as **mixture**: 12
- **false**: 24 errors out of 41 (58.5%)
  - Misclassified as **true**: 1
  - Misclassified as **mixture**: 23
- **mixture**: 5 errors out of 14 (35.7%)
  - Misclassified as **true**: 2
  - Misclassified as **false**: 3

---

### sanity - sanity_nocompression_healthfc_test

#### Class-wise Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| true | 0.800 | 0.364 | 0.500 | 11 |
| false | 0.875 | 0.438 | 0.583 | 16 |
| mixture | 0.118 | 0.667 | 0.200 | 3 |

#### Confusion Matrix (True \ Pred)

| True \ Pred | true | false | mixture |
|---|---|---|---|
| **true** | 4 | 0 | 7 |
| **false** | 1 | 7 | 8 |
| **mixture** | 0 | 1 | 2 |

#### Per-class Error Breakdown

- **true**: 7 errors out of 11 (63.6%)
  - Misclassified as **mixture**: 7
- **false**: 9 errors out of 16 (56.2%)
  - Misclassified as **true**: 1
  - Misclassified as **mixture**: 8
- **mixture**: 1 errors out of 3 (33.3%)
  - Misclassified as **false**: 1

---

## MEDCHANGEQA

### Overall Metrics

| ID | Tag | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) |
|---|---|---|---|---|---|
| BEST1 | BEST1_homo_dense_bm25_medchangeqa | 0.609 | 0.372 | 0.353 | 0.330 |
| BEST2 | BEST2_hetero_dense_bm25_medchangeqa | 0.634 | 0.409 | 0.354 | 0.315 |
| C1 | C1_bm25_flat_medchangeqa | 0.604 | 0.379 | 0.349 | 0.312 |
| C2 | C2_bm25_recency_a0.3_medchangeqa | 0.627 | 0.373 | 0.348 | 0.318 |
| C3 | C3_bm25_recency_a0.7_medchangeqa | 0.591 | 0.284 | 0.318 | 0.272 |
| F2 | F2_qwen14b_recency0.3_with_decomp_medchangeqa | 0.630 | 0.388 | 0.354 | 0.324 |
| G1a | G1a_pubmed_recency_a0.1_medchangeqa_test | 0.639 | 0.381 | 0.340 | 0.286 |
| G1b | G1b_pubmed_recency_a0.2_medchangeqa_test | 0.650 | 0.548 | 0.348 | 0.290 |
| G1c | G1c_pubmed_recency_a0.3_medchangeqa_test | 0.617 | 0.374 | 0.339 | 0.279 |
| G1d | G1d_pubmed_recency_a0.5_medchangeqa_test | 0.672 | 0.556 | 0.350 | 0.298 |
| G1e | G1e_pubmed_recency_a0.7_medchangeqa_test | 0.667 | 0.554 | 0.349 | 0.296 |
| G1f | G1f_pubmed_flat_medchangeqa_test | 0.566 | 0.304 | 0.327 | 0.265 |

### BEST1 - BEST1_homo_dense_bm25_medchangeqa

#### Class-wise Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| supported | 0.629 | 0.885 | 0.735 | 174 |
| refuted | 0.487 | 0.173 | 0.255 | 110 |
| not enough info | 0.000 | 0.000 | 0.000 | 0 |

#### Confusion Matrix (True \ Pred)

| True \ Pred | supported | refuted | not enough info |
|---|---|---|---|
| **supported** | 154 | 20 | 0 |
| **refuted** | 91 | 19 | 0 |
| **not enough info** | 0 | 0 | 0 |

#### Per-class Error Breakdown

- **supported**: 20 errors out of 174 (11.5%)
  - Misclassified as **refuted**: 20
- **refuted**: 91 errors out of 110 (82.7%)
  - Misclassified as **supported**: 91
- **not enough info**: 0 errors out of 0 (0.0%)

---

### BEST2 - BEST2_hetero_dense_bm25_medchangeqa

#### Class-wise Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| supported | 0.637 | 0.955 | 0.764 | 200 |
| refuted | 0.591 | 0.107 | 0.181 | 122 |
| not enough info | 0.000 | 0.000 | 0.000 | 0 |

#### Confusion Matrix (True \ Pred)

| True \ Pred | supported | refuted | not enough info |
|---|---|---|---|
| **supported** | 191 | 9 | 0 |
| **refuted** | 109 | 13 | 0 |
| **not enough info** | 0 | 0 | 0 |

#### Per-class Error Breakdown

- **supported**: 9 errors out of 200 (4.5%)
  - Misclassified as **refuted**: 9
- **refuted**: 109 errors out of 122 (89.3%)
  - Misclassified as **supported**: 109
- **not enough info**: 0 errors out of 0 (0.0%)

---

### C1 - C1_bm25_flat_medchangeqa

#### Class-wise Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| supported | 0.613 | 0.926 | 0.737 | 135 |
| refuted | 0.524 | 0.122 | 0.198 | 90 |
| not enough info | 0.000 | 0.000 | 0.000 | 0 |

#### Confusion Matrix (True \ Pred)

| True \ Pred | supported | refuted | not enough info |
|---|---|---|---|
| **supported** | 125 | 10 | 0 |
| **refuted** | 79 | 11 | 0 |
| **not enough info** | 0 | 0 | 0 |

#### Per-class Error Breakdown

- **supported**: 10 errors out of 135 (7.4%)
  - Misclassified as **refuted**: 10
- **refuted**: 79 errors out of 90 (87.8%)
  - Misclassified as **supported**: 79
- **not enough info**: 0 errors out of 0 (0.0%)

---

### C2 - C2_bm25_recency_a0.3_medchangeqa

#### Class-wise Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| supported | 0.643 | 0.920 | 0.757 | 137 |
| refuted | 0.476 | 0.125 | 0.198 | 80 |
| not enough info | 0.000 | 0.000 | 0.000 | 0 |

#### Confusion Matrix (True \ Pred)

| True \ Pred | supported | refuted | not enough info |
|---|---|---|---|
| **supported** | 126 | 11 | 0 |
| **refuted** | 70 | 10 | 0 |
| **not enough info** | 0 | 0 | 0 |

#### Per-class Error Breakdown

- **supported**: 11 errors out of 137 (8.0%)
  - Misclassified as **refuted**: 11
- **refuted**: 70 errors out of 80 (87.5%)
  - Misclassified as **supported**: 70
- **not enough info**: 0 errors out of 0 (0.0%)

---

### C3 - C3_bm25_recency_a0.7_medchangeqa

#### Class-wise Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| supported | 0.620 | 0.907 | 0.737 | 108 |
| refuted | 0.231 | 0.048 | 0.079 | 63 |
| not enough info | 0.000 | 0.000 | 0.000 | 0 |

#### Confusion Matrix (True \ Pred)

| True \ Pred | supported | refuted | not enough info |
|---|---|---|---|
| **supported** | 98 | 10 | 0 |
| **refuted** | 60 | 3 | 0 |
| **not enough info** | 0 | 0 | 0 |

#### Per-class Error Breakdown

- **supported**: 10 errors out of 108 (9.3%)
  - Misclassified as **refuted**: 10
- **refuted**: 60 errors out of 63 (95.2%)
  - Misclassified as **supported**: 60
- **not enough info**: 0 errors out of 0 (0.0%)

---

### F2 - F2_qwen14b_recency0.3_with_decomp_medchangeqa

#### Class-wise Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| supported | 0.641 | 0.926 | 0.758 | 135 |
| refuted | 0.524 | 0.136 | 0.216 | 81 |
| not enough info | 0.000 | 0.000 | 0.000 | 0 |

#### Confusion Matrix (True \ Pred)

| True \ Pred | supported | refuted | not enough info |
|---|---|---|---|
| **supported** | 125 | 10 | 0 |
| **refuted** | 70 | 11 | 0 |
| **not enough info** | 0 | 0 | 0 |

#### Per-class Error Breakdown

- **supported**: 10 errors out of 135 (7.4%)
  - Misclassified as **refuted**: 10
- **refuted**: 70 errors out of 81 (86.4%)
  - Misclassified as **supported**: 70
- **not enough info**: 0 errors out of 0 (0.0%)

---

### G1a - G1a_pubmed_recency_a0.1_medchangeqa_test

#### Class-wise Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| supported | 0.644 | 0.974 | 0.776 | 39 |
| refuted | 0.500 | 0.045 | 0.083 | 22 |
| not enough info | 0.000 | 0.000 | 0.000 | 0 |

#### Confusion Matrix (True \ Pred)

| True \ Pred | supported | refuted | not enough info |
|---|---|---|---|
| **supported** | 38 | 1 | 0 |
| **refuted** | 21 | 1 | 0 |
| **not enough info** | 0 | 0 | 0 |

#### Per-class Error Breakdown

- **supported**: 1 errors out of 39 (2.6%)
  - Misclassified as **refuted**: 1
- **refuted**: 21 errors out of 22 (95.5%)
  - Misclassified as **supported**: 21
- **not enough info**: 0 errors out of 0 (0.0%)

---

### G1b - G1b_pubmed_recency_a0.2_medchangeqa_test

#### Class-wise Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| supported | 0.644 | 1.000 | 0.784 | 38 |
| refuted | 1.000 | 0.045 | 0.087 | 22 |
| not enough info | 0.000 | 0.000 | 0.000 | 0 |

#### Confusion Matrix (True \ Pred)

| True \ Pred | supported | refuted | not enough info |
|---|---|---|---|
| **supported** | 38 | 0 | 0 |
| **refuted** | 21 | 1 | 0 |
| **not enough info** | 0 | 0 | 0 |

#### Per-class Error Breakdown

- **supported**: 0 errors out of 38 (0.0%)
- **refuted**: 21 errors out of 22 (95.5%)
  - Misclassified as **supported**: 21
- **not enough info**: 0 errors out of 0 (0.0%)

---

### G1c - G1c_pubmed_recency_a0.3_medchangeqa_test

#### Class-wise Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| supported | 0.621 | 0.973 | 0.758 | 37 |
| refuted | 0.500 | 0.043 | 0.080 | 23 |
| not enough info | 0.000 | 0.000 | 0.000 | 0 |

#### Confusion Matrix (True \ Pred)

| True \ Pred | supported | refuted | not enough info |
|---|---|---|---|
| **supported** | 36 | 1 | 0 |
| **refuted** | 22 | 1 | 0 |
| **not enough info** | 0 | 0 | 0 |

#### Per-class Error Breakdown

- **supported**: 1 errors out of 37 (2.7%)
  - Misclassified as **refuted**: 1
- **refuted**: 22 errors out of 23 (95.7%)
  - Misclassified as **supported**: 22
- **not enough info**: 0 errors out of 0 (0.0%)

---

### G1d - G1d_pubmed_recency_a0.5_medchangeqa_test

#### Class-wise Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| supported | 0.667 | 1.000 | 0.800 | 38 |
| refuted | 1.000 | 0.050 | 0.095 | 20 |
| not enough info | 0.000 | 0.000 | 0.000 | 0 |

#### Confusion Matrix (True \ Pred)

| True \ Pred | supported | refuted | not enough info |
|---|---|---|---|
| **supported** | 38 | 0 | 0 |
| **refuted** | 19 | 1 | 0 |
| **not enough info** | 0 | 0 | 0 |

#### Per-class Error Breakdown

- **supported**: 0 errors out of 38 (0.0%)
- **refuted**: 19 errors out of 20 (95.0%)
  - Misclassified as **supported**: 19
- **not enough info**: 0 errors out of 0 (0.0%)

---

### G1e - G1e_pubmed_recency_a0.7_medchangeqa_test

#### Class-wise Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| supported | 0.661 | 1.000 | 0.796 | 39 |
| refuted | 1.000 | 0.048 | 0.091 | 21 |
| not enough info | 0.000 | 0.000 | 0.000 | 0 |

#### Confusion Matrix (True \ Pred)

| True \ Pred | supported | refuted | not enough info |
|---|---|---|---|
| **supported** | 39 | 0 | 0 |
| **refuted** | 20 | 1 | 0 |
| **not enough info** | 0 | 0 | 0 |

#### Per-class Error Breakdown

- **supported**: 0 errors out of 39 (0.0%)
- **refuted**: 20 errors out of 21 (95.2%)
  - Misclassified as **supported**: 20
- **not enough info**: 0 errors out of 0 (0.0%)

---

### G1f - G1f_pubmed_flat_medchangeqa_test

#### Class-wise Metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| supported | 0.580 | 0.935 | 0.716 | 31 |
| refuted | 0.333 | 0.045 | 0.080 | 22 |
| not enough info | 0.000 | 0.000 | 0.000 | 0 |

#### Confusion Matrix (True \ Pred)

| True \ Pred | supported | refuted | not enough info |
|---|---|---|---|
| **supported** | 29 | 2 | 0 |
| **refuted** | 21 | 1 | 0 |
| **not enough info** | 0 | 0 | 0 |

#### Per-class Error Breakdown

- **supported**: 2 errors out of 31 (6.5%)
  - Misclassified as **refuted**: 2
- **refuted**: 21 errors out of 22 (95.5%)
  - Misclassified as **supported**: 21
- **not enough info**: 0 errors out of 0 (0.0%)

---

