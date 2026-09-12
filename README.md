# EAAI v25 reproducibility package

This archive contains the code, metadata, held-out per-record predictions, aggregate results, environment record, and SHA-256 hashes for the target-domain adaptation experiment reported in the manuscript.

## Data sources

The SCA bearing dataset V1 is publicly available from Mendeley Data (DOI: 10.17632/tdn96mkkpt.1; CC BY 4.0). The archive does not redistribute the raw public data; download it from the DOI landing page and verify the files against `raw_file_hashes/sha256.json`.

## Frozen protocol

The strict field retraining protocol uses only case 1–3 `test.mat` files. DS and FS records from each case share one chronological split. The first 56% is used to train a fixed DirectMLP, the next 14% tunes the threshold, and the final 30% is an untouched holdout. Post-replacement `case1_train.mat` is retained only for chronology auditing and is excluded from the main result. Five seeds (11, 22, 33, 44, 55) are retained. `results/strict_case_retraining/holdout_predictions_seed_*.json` contains probabilities, labels, timestamps, case, sensor position, and binary predictions for every held-out record.

## Reported result

The strict result is Macro-F1 = 0.5116 (SD 0.0245), Jaccard = 0.8512, exact set match = 0.8512, healthy-record false-alarm rate = 0.0034, fault-record recall = 0.5040, and device-case event recall = 2/3. The two detected case events have lead times of 37.4 and 8.8 days, respectively. The result supports only retrospective conditional utility after same-case supervised retraining; it does not establish cross-device or universal field effectiveness.

## Archive step before submission

The public archive is available at <https://github.com/Jhson14678/eaai-bearing-v25-reproducibility>. The manuscript Data Availability statement cites this URL.

## Re-run entry point

Install the pinned packages in `requirements.txt`, download the SCA V1 public files, and set `SCA_FIELD_ROOT` to the directory containing `case1_test.mat`, `case2_test.mat`, and `case3_test.mat`. Run `python code/run_sca_strict_case_retraining.py`. Set `SCA_OUTPUT_ROOT` to choose the result directory. The script writes the five-seed predictions and `strict_results.json`.
