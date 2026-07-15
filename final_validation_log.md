# Final Validation Log

Generated: 2026-07-15T14:36:56

This file records the final, frozen, end-to-end pipeline results after CV routing, VLM fallback, indeterminate handling, and the final PASS safety guardrail.

## 1. Evaluation Summary


| Metric | Value |
|---|---:|
| Evaluation images | 250 |
| End-to-end accuracy | 0.4960 |
| Macro F1 | 0.5189 |
| Weighted F1 | 0.4953 |
| Final PASS decisions | 113 |
| PASS rate | 45.20% |
| Images kept out of PASS | 137 |
| Non-PASS rate | 54.80% |
| FLAG decisions | 57 |
| DENOISE decisions | 64 |
| BLOCK decisions | 13 |
| Final manual-review decisions | 3 |
| Manual-review / indeterminate rate | 1.20% |
| Review or remediation decisions | 124 |
| Review or remediation rate | 49.60% |

## 2. Independent Per-Class Metrics

|   Class ID | Class                |   Precision |   Recall |     F1 |   Support |
|-----------:|:---------------------|------------:|---------:|-------:|----------:|
|          1 | Clear and Readable   |      0.4513 |   0.6711 | 0.5397 |        76 |
|          2 | Content Got Cut      |      0.5088 |   0.3816 | 0.4361 |        76 |
|          3 | Unclear / Unreadable |      0.5156 |   0.4342 | 0.4714 |        76 |
|          4 | Random Image         |      0.8462 |   0.5    | 0.6286 |        22 |

## 3. Total Pipeline Confusion Matrix

Rows are true classes. Columns are final pipeline predictions.

|                              |   Pred 1: Clear and Readable |   Pred 2: Content Got Cut |   Pred 3: Unclear / Unreadable |   Pred 4: Random Image |   Pred 0: Indeterminate |
|:-----------------------------|-----------------------------:|--------------------------:|-------------------------------:|-----------------------:|------------------------:|
| True 1: Clear and Readable   |                           51 |                        15 |                              8 |                      1 |                       1 |
| True 2: Content Got Cut      |                           26 |                        29 |                             21 |                      0 |                       0 |
| True 3: Unclear / Unreadable |                           30 |                        12 |                             33 |                      1 |                       0 |
| True 4: Random Image         |                            6 |                         1 |                              2 |                     11 |                       2 |
| True 0: Indeterminate        |                            0 |                         0 |                              0 |                      0 |                       0 |

## 4. Final Routing Outcomes by True Class

|   Class ID | True class           | PASS       | FLAG       | DENOISE    | BLOCK      | MANUAL_REVIEW   |   Support |
|-----------:|:---------------------|:-----------|:-----------|:-----------|:-----------|:----------------|----------:|
|          1 | Clear and Readable   | 51 (67.1%) | 15 (19.7%) | 8 (10.5%)  | 1 (1.3%)   | 1 (1.3%)        |        76 |
|          2 | Content Got Cut      | 26 (34.2%) | 29 (38.2%) | 21 (27.6%) | 0 (0.0%)   | 0 (0.0%)        |        76 |
|          3 | Unclear / Unreadable | 30 (39.5%) | 12 (15.8%) | 33 (43.4%) | 1 (1.3%)   | 0 (0.0%)        |        76 |
|          4 | Random Image         | 6 (27.3%)  | 1 (4.5%)   | 2 (9.1%)   | 11 (50.0%) | 2 (9.1%)        |        22 |

## 5. Classes 2 and 3 Safety Against Unsafe PASS

|   Class ID | Class                |   Support |   Incorrectly sent to PASS |   Kept out of PASS | Kept out of PASS rate   |
|-----------:|:---------------------|----------:|---------------------------:|-------------------:|:------------------------|
|          2 | Content Got Cut      |        76 |                         26 |                 50 | 65.8%                   |
|          3 | Unclear / Unreadable |        76 |                         30 |                 46 | 60.5%                   |

## 6. Class 4 Filter Bypass

| Metric | Value |
|---|---:|
| Class 4 support | 22 |
| Class 4 images routed to PASS | 6 |
| Class 4 bypass rate | 27.27% |
| Requirement: bypass count must equal 0 | **FAIL** |

### Class 4 Bypass Records

| image_path                               |   final_prediction | decision_source   | cv_route     |   document_score |   crop_score |   quality_score |   confidence_score | safety_gate_reason   |
|:-----------------------------------------|-------------------:|:------------------|:-------------|-----------------:|-------------:|----------------:|-------------------:|:---------------------|
| class4_nonDocument/hard_negatives/1.jpg  |                  1 | cv                | cv_class1    |         0.757843 |     0.132154 |        0.436085 |             nan    | <NA>                 |
| class4_nonDocument/hard_negatives/10.jpg |                  1 | vlm               | vlm_fallback |         0.353418 |     0.608791 |        0.466286 |               0.95 | <NA>                 |
| class4_nonDocument/hard_negatives/2.jpg  |                  1 | vlm               | vlm_fallback |         0.387209 |     0.195167 |        0.518002 |               0.95 | <NA>                 |
| class4_nonDocument/hard_negatives/4.jpg  |                  1 | vlm               | vlm_fallback |         0.431621 |     0.176749 |        0.416994 |               0.95 | <NA>                 |
| class4_nonDocument/hard_negatives/7.jpg  |                  1 | vlm               | vlm_fallback |         0.353225 |     0.398437 |        0.555818 |               0.95 | <NA>                 |
| class4_nonDocument/indoors/2.jpg         |                  1 | vlm               | vlm_fallback |         0.519816 |     0.527243 |        0.595253 |               0.85 | <NA>                 |

## 7. Indeterminate Bucket

| Metric | Value |
|---|---:|
| Indeterminate bucket size | 3 |
| Percentage of evaluation set | 1.20% |

### True-Class Composition

|   Class ID | True class           |   Images | Percentage of bucket   |
|-----------:|:---------------------|---------:|:-----------------------|
|          1 | Clear and Readable   |        1 | 33.3%                  |
|          2 | Content Got Cut      |        0 | 0.0%                   |
|          3 | Unclear / Unreadable |        0 | 0.0%                   |
|          4 | Random Image         |        2 | 66.7%                  |

## 8. Confusion Breakdown by Ordered Class Pair

This table acts as the pair-level confusion log. Each row records one true-class to predicted-class pair independently.

|   True class ID | True class           |   Predicted class ID | Predicted class      |   Count | Percentage of true class   | Correct pair   |
|----------------:|:---------------------|---------------------:|:---------------------|--------:|:---------------------------|:---------------|
|               1 | Clear and Readable   |                    1 | Clear and Readable   |      51 | 67.1%                      | True           |
|               1 | Clear and Readable   |                    2 | Content Got Cut      |      15 | 19.7%                      | False          |
|               1 | Clear and Readable   |                    3 | Unclear / Unreadable |       8 | 10.5%                      | False          |
|               1 | Clear and Readable   |                    4 | Random Image         |       1 | 1.3%                       | False          |
|               1 | Clear and Readable   |                    0 | Indeterminate        |       1 | 1.3%                       | False          |
|               2 | Content Got Cut      |                    1 | Clear and Readable   |      26 | 34.2%                      | False          |
|               2 | Content Got Cut      |                    2 | Content Got Cut      |      29 | 38.2%                      | True           |
|               2 | Content Got Cut      |                    3 | Unclear / Unreadable |      21 | 27.6%                      | False          |
|               3 | Unclear / Unreadable |                    1 | Clear and Readable   |      30 | 39.5%                      | False          |
|               3 | Unclear / Unreadable |                    2 | Content Got Cut      |      12 | 15.8%                      | False          |
|               3 | Unclear / Unreadable |                    3 | Unclear / Unreadable |      33 | 43.4%                      | True           |
|               3 | Unclear / Unreadable |                    4 | Random Image         |       1 | 1.3%                       | False          |
|               4 | Random Image         |                    1 | Clear and Readable   |       6 | 27.3%                      | False          |
|               4 | Random Image         |                    2 | Content Got Cut      |       1 | 4.5%                       | False          |
|               4 | Random Image         |                    3 | Unclear / Unreadable |       2 | 9.1%                       | False          |
|               4 | Random Image         |                    4 | Random Image         |      11 | 50.0%                      | True           |
|               4 | Random Image         |                    0 | Indeterminate        |       2 | 9.1%                       | False          |


## 9. Narrative Summary of Under- and Over-Flagging

### Class 1 — Clear and Readable

Class 1 recall was 0.6711. The most frequent non-PASS outcome was `FLAG`. This indicates that the pipeline tends to under-pass some clear documents when crop or quality evidence is ambiguous.

__**Over Flagging Failure Example (Ambiguous border makes it look cropped)**__
![Class1Failure](report_imgs/class1_failure.png)

### Class 2 — Content Got Cut

Class 2 recall was 0.3816. 26 cropped documents were incorrectly routed to PASS. Other errors were generally routed toward FLAG, DENOISE, or manual review, reflecting the intended safety preference for catching suspected cropping even at the cost of false positives.

__**Under Flagging Failure Example (Images still readable)**__
![Class2Failure](report_imgs/class2_failure_weak_cropping.png)

### Class 3 — Unclear / Unreadable

Class 3 recall was 0.4342. 30 degraded documents were incorrectly routed to PASS. The dominant boundary remains moderate degradation where some visible content is still readable, making the OCR-readiness decision ambiguous.

__**Under Flagging Failure Example (Images still readable)**__
![Class3Failure](report_imgs/class3_failure_weak_degradation.png)

### Class 4 — Random Image

Class 4 recall was 0.5000. The final Class 4 filter bypass count was 6. Remaining bypasses should be inspected as high-priority safety failures, particularly when non-document images contain text, page-like layouts, screens, publications, or other document-like structure.

__**UnderFlagging Failure Exmaples (Images look like documents, hard negatives)**__
![Class4Failur](report_imgs/class4_failure_hard_negatives.png)
