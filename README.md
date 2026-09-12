# EAAI v25 reproducibility package

This archive contains the code, metadata, held-out per-record predictions, aggregate results, environment record, and SHA-256 hashes for the target-domain adaptation experiment reported in the manuscript.

## Data sources

The SCA bearing dataset V1 is publicly available from Mendeley Data (DOI: 10.17632/tdn96mkkpt.1; CC BY 4.0). The archive does not redistribute the raw public data; download it from the DOI landing page and verify the files against `raw_file_hashes/sha256.json`.

## Frozen protocol

The field retraining protocol trains a fixed DirectMLP architecture using the first 70% of each SCA test group in chronological order plus case 1 post-replacement healthy train records. The final 30% is an untouched time-out holdout. Thresholds are selected from the adaptation segment. Five seeds (11, 22, 33, 44, 55) are retained. `results/retraining/holdout_predictions_seed_*.json` contains probabilities, labels, timestamps, groups, and binary predictions for every held-out record.

## Reported result

Mean Macro-F1 = 0.5033 (SD 0.0263), Jaccard = 0.8319, exact set match = 0.8319, healthy-record false-alarm rate = 0, fault-record recall = 0.5453, fault-event recall = 1.0 across three held-out fault groups, and mean warning lead = 28.5 days. The result supports conditional utility after target-site retraining; the small number of field groups still does not establish universal field effectiveness.

## Archive step before submission

The public archive is available at <https://github.com/Jhson14678/eaai-bearing-v25-reproducibility>. The manuscript Data Availability statement cites this URL.
