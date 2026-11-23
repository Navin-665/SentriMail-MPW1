# E0 Baseline Evaluation Report

Generated from `wsp_runs/e0/{val,test}/e0_metrics.json`. Protocol: {'category_lengths': [512, 128], 'csat_len': 128, 'autocast': 'fp16', 'mode': 'model.eval() + torch.inference_mode()', 'batch_size': 128}.

## Category routing - val

| Checkpoint | Len | Acc | Macro F1 | Weighted F1 | NLL | ECE | Brier | Mean conf | Conf<25% | Returns rate |
|---|---|---|---|---|---|---|---|---|---|---|
| classifier_best | 512 | 79.34% | 0.4197 | 0.7907 | 0.599 | 0.0492 | 0.292 | 0.843 | 0.15% | 0.64% |
| classifier_best | 128 | 77.13% | 0.3995 | 0.7685 | 0.668 | 0.0549 | 0.323 | 0.826 | 0.17% | 0.64% |
| multitask_best | 512 | 74.34% | 0.3280 | 0.7371 | 0.731 | 0.0441 | 0.355 | 0.787 | 0.37% | 0.71% |
| multitask_best | 128 | 73.57% | 0.3239 | 0.7296 | 0.765 | 0.0536 | 0.367 | 0.789 | 0.38% | 0.71% |
| m10_3_sft_best | 512 | 39.66% | 0.0797 | 0.3884 | 2.014 | 0.1652 | 0.779 | 0.231 | 63.33% | 4.82% |
| m10_3_sft_best | 128 | 42.50% | 0.0884 | 0.4204 | 2.109 | 0.2377 | 0.799 | 0.187 | 80.19% | 4.82% |

## CSAT - val

| Checkpoint | Exact | Within±1 | MAE | RMSE | Macro F1 | Weighted F1 | ECE | Mean conf | Pred dist (1..5) |
|---|---|---|---|---|---|---|---|---|---|
| classifier_best | SKIPPED: CSAT head untrained in this checkpoint |||||||||
| multitask_best | 71.50% | 79.15% | 0.795 | 1.638 | 0.2989 | 0.6605 | 0.0719 | 0.779 | [249, 0, 1, 5, 594] |
| m10_3_sft_best | 66.31% | 72.08% | 1.046 | 1.924 | 0.2790 | 0.6210 | 0.0921 | 0.579 | [371, 0, 2, 0, 476] |

## Paired comparisons - val (B - A; 1000x paired bootstrap 95% CI; exact McNemar)

| Comparison | ΔAcc | 95% CI | ΔMacro F1 | 95% CI | McNemar (A✗B✓ / A✓B✗, p) |
|---|---|---|---|---|---|
| category_len512: multitask_best - classifier_best | -5.00 pp | [-5.21, -4.80] | -0.0917 | [-0.1013, -0.0863] | 5264 / 11412, p=0 |
| category_len512: m10_3_sft_best - multitask_best | -34.67 pp | [-34.96, -34.37] | -0.2483 | [-0.2666, -0.2456] | 5086 / 47722, p=0 |
| category_len512: m10_3_sft_best - classifier_best | -39.67 pp | [-39.97, -39.38] | -0.3400 | [-0.3653, -0.3345] | 4789 / 53573, p=0 |
| category_len128: multitask_best - classifier_best | -3.56 pp | [-3.76, -3.37] | -0.0756 | [-0.0848, -0.0700] | 5675 / 10050, p=4.5e-270 |
| category_len128: m10_3_sft_best - multitask_best | -31.07 pp | [-31.36, -30.77] | -0.2354 | [-0.2532, -0.2321] | 5040 / 43246, p=0 |
| category_len128: m10_3_sft_best - classifier_best | -34.63 pp | [-34.95, -34.34] | -0.3111 | [-0.3353, -0.3056] | 4966 / 47547, p=0 |
| csat_len128: m10_3_sft_best - multitask_best | -5.18 pp | [-7.77, -2.82] | -0.0199 | [-0.0317, -0.0080] | 37 / 81, p=6.3e-05 |

## Category routing - test

| Checkpoint | Len | Acc | Macro F1 | Weighted F1 | NLL | ECE | Brier | Mean conf | Conf<25% | Returns rate |
|---|---|---|---|---|---|---|---|---|---|---|
| classifier_best | 512 | 79.15% | 0.4035 | 0.7887 | 0.605 | 0.0510 | 0.294 | 0.842 | 0.16% | 0.66% |
| classifier_best | 128 | 76.90% | 0.3847 | 0.7661 | 0.676 | 0.0577 | 0.325 | 0.827 | 0.19% | 0.66% |
| multitask_best | 512 | 73.99% | 0.3168 | 0.7336 | 0.740 | 0.0482 | 0.358 | 0.788 | 0.37% | 0.71% |
| multitask_best | 128 | 73.29% | 0.3131 | 0.7268 | 0.774 | 0.0569 | 0.370 | 0.790 | 0.38% | 0.71% |
| m10_3_sft_best | 512 | 39.59% | 0.0777 | 0.3883 | 2.020 | 0.1650 | 0.780 | 0.231 | 63.43% | 4.79% |
| m10_3_sft_best | 128 | 42.27% | 0.0859 | 0.4181 | 2.112 | 0.2355 | 0.800 | 0.187 | 80.10% | 4.79% |

## CSAT - test

| Checkpoint | Exact | Within±1 | MAE | RMSE | Macro F1 | Weighted F1 | ECE | Mean conf | Pred dist (1..5) |
|---|---|---|---|---|---|---|---|---|---|
| classifier_best | SKIPPED: CSAT head untrained in this checkpoint |||||||||
| multitask_best | 73.56% | 81.72% | 0.732 | 1.576 | 0.3169 | 0.6875 | 0.0791 | 0.794 | [250, 0, 0, 9, 611] |
| m10_3_sft_best | 66.90% | 73.68% | 1.030 | 1.919 | 0.2778 | 0.6282 | 0.0777 | 0.594 | [356, 0, 0, 0, 514] |

## Paired comparisons - test (B - A; 1000x paired bootstrap 95% CI; exact McNemar)

| Comparison | ΔAcc | 95% CI | ΔMacro F1 | 95% CI | McNemar (A✗B✓ / A✓B✗, p) |
|---|---|---|---|---|---|
| category_len512: multitask_best - classifier_best | -5.15 pp | [-5.37, -4.96] | -0.0868 | [-0.0980, -0.0817] | 5225 / 11573, p=0 |
| category_len512: m10_3_sft_best - multitask_best | -34.41 pp | [-34.71, -34.10] | -0.2391 | [-0.2620, -0.2364] | 5188 / 47589, p=0 |
| category_len512: m10_3_sft_best - classifier_best | -39.56 pp | [-39.88, -39.23] | -0.3259 | [-0.3566, -0.3213] | 4726 / 53475, p=0 |
| category_len128: multitask_best - classifier_best | -3.61 pp | [-3.81, -3.42] | -0.0715 | [-0.0813, -0.0663] | 5660 / 10108, p=1.9e-278 |
| category_len128: m10_3_sft_best - multitask_best | -31.02 pp | [-31.32, -30.71] | -0.2273 | [-0.2488, -0.2241] | 5129 / 43352, p=0 |
| category_len128: m10_3_sft_best - classifier_best | -34.63 pp | [-34.95, -34.30] | -0.2988 | [-0.3275, -0.2941] | 5072 / 47743, p=0 |
| csat_len128: m10_3_sft_best - multitask_best | -6.67 pp | [-9.20, -4.25] | -0.0390 | [-0.0579, -0.0219] | 36 / 94, p=3.7e-07 |

## Per-class F1 - test, len 512

| Class | Support | classifier_best | multitask_best | m10_3_sft_best | m10_3 pred count |
|---|---|---|---|---|---|
| App/website | 1 | 0.000 | 0.000 | 0.000 | 0 |
| Bank account or service | 768 | 0.620 | 0.420 | 0.000 | 0 |
| Cancellation | 24 | 0.000 | 0.000 | 0.000 | 0 |
| Checking or savings account | 9199 | 0.831 | 0.791 | 0.294 | 50953 |
| Consumer Loan | 477 | 0.522 | 0.268 | 0.000 | 0 |
| Credit card | 5756 | 0.669 | 0.595 | 0.038 | 934 |
| Credit card or prepaid card | 5151 | 0.647 | 0.604 | 0.247 | 3916 |
| Credit reporting | 1530 | 0.579 | 0.342 | 0.000 | 0 |
| Credit reporting or other personal consumer reports | 33547 | 0.832 | 0.779 | 0.616 | 26175 |
| Credit reporting, credit repair services, or other personal consumer reports | 26627 | 0.777 | 0.709 | 0.489 | 25838 |
| Debt collection | 18488 | 0.813 | 0.783 | 0.476 | 8628 |
| Debt or credit management | 262 | 0.361 | 0.000 | 0.000 | 0 |
| Feedback | 24 | 0.000 | 0.000 | 0.000 | 0 |
| Money transfer, virtual currency, or money service | 4841 | 0.802 | 0.766 | 0.149 | 626 |
| Money transfers | 71 | 0.537 | 0.080 | 0.000 | 0 |
| Mortgage | 7320 | 0.923 | 0.910 | 0.034 | 143 |
| Offers & Cashback | 6 | 0.000 | 0.000 | 0.000 | 0 |
| Onboarding related | 1 | 0.000 | 0.000 | 0.000 | 0 |
| Order Related | 244 | 0.186 | 0.008 | 0.000 | 0 |
| Other financial service | 16 | 0.000 | 0.000 | 0.000 | 0 |
| Others | 2 | 0.000 | 0.000 | 0.000 | 0 |
| Payday loan | 100 | 0.361 | 0.000 | 0.000 | 0 |
| Payday loan, title loan, or personal loan | 838 | 0.478 | 0.354 | 0.000 | 0 |
| Payday loan, title loan, personal loan, or advance loan | 904 | 0.545 | 0.423 | 0.000 | 0 |
| Payments related | 22 | 0.000 | 0.000 | 0.000 | 0 |
| Prepaid card | 592 | 0.576 | 0.462 | 0.049 | 19 |
| Product Queries | 40 | 0.000 | 0.000 | 0.000 | 0 |
| Refund Related | 39 | 0.000 | 0.000 | 0.000 | 0 |
| Returns | 431 | 0.643 | 0.631 | 0.131 | 5897 |
| Shopzilla Related | 36 | 0.000 | 0.000 | 0.000 | 0 |
| Student loan | 3240 | 0.873 | 0.847 | 0.010 | 35 |
| Vehicle loan or lease | 2630 | 0.743 | 0.680 | 0.030 | 65 |
| Virtual currency | 2 | 0.000 | 0.000 | 0.000 | 0 |

## Integrity

- Files hashed before and after: 16; changed: **0** 
- In-memory weight fingerprint identical before/after evaluation for every checkpoint (asserted).
- Split checks: {"classification": {"id_overlap": {"train&val": 0, "train&test": 0, "val&test": 0}, "text_overlap": {"train&val": 0, "train&test": 0, "val&test": 0}}, "csat": {"id_overlap": {"train&val": 0, "train&test": 0, "val&test": 0}, "text_overlap": {"train&val": 0, "train&test": 0, "val&test": 0}}}
- Cross-task eval/train exact-text overlaps: all 0 (zero means none).
- Pretraining-corpus overlap: {"classification_train": {"n": 2219168, "in_pretrain_train": 2219168, "in_pretrain_val": 0, "in_pretrain_test": 0}, "classification_val": {"n": 122961, "in_pretrain_train": 0, "in_pretrain_val": 122961, "in_pretrain_test": 0}, "classification_test": {"n": 123229, "in_pretrain_train": 0, "in_pretrain_val": 0, "in_pretrain_test": 123229}, "csat_train": {"n": 15217, "in_pretrain_train": 15217, "in_pretrain_val": 0, "in_pretrain_test": 0}, "csat_val": {"n": 849, "in_pretrain_train": 0, "in_pretrain_val": 849, "in_pretrain_test": 0}, "csat_test": {"n": 870, "in_pretrain_train": 0, "in_pretrain_val": 0, "in_pretrain_test": 870}}

### SHA256

| File | SHA256 | Bytes |
|---|---|---|
| checkpoints/classification/classifier_best.pt | `b6276824c6bb17c342552a8aa9f5a5e056e9c77c06e495237f43458e18b748d1` | 105,275,550 |
| checkpoints/multitask/multitask_best.pt | `63a22b005be7a4cbe411859971e0ad76b3b4ec116870e6c7404485822798dec5` | 139,476,835 |
| checkpoints/m10_3_sft/m10_3_sft_best.pt | `2b0e40346eeb37135ae3a67603ef3a482b576dbb733f657d880834ddca6d6960` | 138,345,033 |
| data/supervised/classification_train.parquet | `2bb10e6c34a9e079795b6889f6b5a4aa05f07376afb3169784aa41438745615d` | 1,211,083,695 |
| data/supervised/classification_val.parquet | `070e4baa8ce297ad33fc996f6570240430fcacde64825d1ccb87b978bd84744b` | 72,795,700 |
| data/supervised/classification_test.parquet | `acadb7160ec6a90bb8094ffce8cd53b23559888b9a98b4c91238acd8411397d5` | 72,832,444 |
| data/supervised/classification_labels.json | `1720a5aaab14234b781a66f20e043d723c638cf24153376e9affb6cb2c555e00` | 2,377 |
| data/supervised/csat_train.parquet | `47d859ad80208b00267354bd0acbc204d324c4bf470857561eefc487cc01fc5c` | 1,260,185 |
| data/supervised/csat_val.parquet | `f4514e47ee4f79fa479ed10dfcd80e61e2e9c160eecbce8c9c265dc6fcbe5d26` | 78,100 |
| data/supervised/csat_test.parquet | `3217f60e3fc01de06221dae7ae5b7c35cf33b8241b25b16a7da14cbce645ba33` | 77,002 |
| data/supervised/dialogue_train.parquet | `118c4aa77907a0885258f0595e217c65939e8a7bea28a7847e0a797fa69bb75b` | 140,804,516 |
| data/supervised/clean_dialogue_train.parquet | `67a886f753e9951d43a74b4c031b00161f2484ca26aca2dd6e179f0ffd026a82` | 61,416,595 |
| data/supervised/m10_3_dialogue_train.parquet | `ee9ea706ad530fd2a80e677981aa30ace63c20b23502feb82579b7ac6b062a88` | 53,943,770 |
| checkpoints/config.json | `6bd20e6085d962d268bb716cba976aa65da55ac5e1032d95a46019550f211fca` | 567 |
| tokenizer/sentritokenizer_v1/tokenizer.json | `528dca6bc169c41313c994d17cc45032be50cd477e845c6e7973e2d22247f6f7` | 1,126,521 |
| tokenizer/sentritokenizer_v1/tokenizer_config.json | `1101dd9d914d617d6b6503678fe380e0b21752b5f587def068d176df47354a6e` | 2,505 |
