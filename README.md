# EAAI reproducibility package (v30 strict protocol)

This archive contains the strict case-level target-domain replay code, metadata, held-out per-record predictions, aggregate results, alarm-persistence audit, environment record, and SHA-256 hashes. Raw public data are not redistributed.

## Data sources

SCA bearing dataset V1: Mendeley Data DOI `10.17632/tdn96mkkpt.1` (CC BY 4.0). Download the files from the DOI landing page and verify `raw_file_hashes/sha256.json`.

## Frozen protocol

Only case 1–3 `test.mat` files are used; post-replacement `case1_train.mat` is excluded. DS and FS records are merged within each case and sorted by timestamp. Elapsed-time boundaries at 56% and 70% define train, threshold-validation and untouched holdout segments. The default feature is a native-rate, band-limited order-envelope representation: no high-frequency reconstruction is performed, and orders above the measurable Nyquist guard band are zero-filled. Set `SCA_FEATURE_MODE=upsample` only for the common-grid sensitivity control. SCA has no HUST load labels, so the condition vector is fixed to a constant.

## Results

Under the native-rate primary protocol, DirectMLP gives Macro-F1 = 0.5251 (SD 0.0072), Jaccard = 0.8819, exact set match = 0.8819, healthy-record false-alarm rate = 0.0094, fault-record recall = 0.6263, and case-level fault-event recall = 2/3. The detected case events have lead times of 37.4 and 26.0 days. These are retrospective same-case supervised replays, not universal field validation.

The proposed non-negative candidate under the identical protocol gives Macro-F1 = 0.4577 (SD 0.1308), Jaccard = 0.8620, healthy-record false-alarm rate = 0.0197, fault-record recall = 0.5838, and mean case-event recall 0.60 across five seeds; it does not show field superiority.

## Alarm persistence audit

`code/evaluate_alarm_persistence.py` applies post-hoc persistence to frozen holdout predictions. Consecutive positives must be within 24 h. Case-event recall remains 2/3 for k=1, 2 and 3. Mean healthy sensor-group event FAR is 0.3333 for k=1 and k=2, and 0 for k=3. This is a sensitivity analysis, not a pre-registered deployment operating point.

## Re-run

Install `requirements.txt`, set `SCA_FIELD_ROOT` to the directory containing `case1_test.mat`, `case2_test.mat` and `case3_test.mat`, and run:

```bash
python code/run_sca_strict_case_retraining.py
```

Set `SCA_OUTPUT_ROOT` to choose the result directory. The default is native-rate features and elapsed-time splitting. The archive is available at <https://github.com/Jhson14678/eaai-bearing-v25-reproducibility> (release `v31.2` (master commit `3d117e8`)).
