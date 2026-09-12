# EAAI v25 reproducibility package

This archive contains the code, metadata, held-out per-record predictions, aggregate results, environment record, and SHA-256 hashes for the target-domain adaptation experiment reported in the manuscript.

## Data sources

The SCA bearing dataset V1 is publicly available from Mendeley Data (DOI: 10.17632/tdn96mkkpt.1; CC BY 4.0). The archive does not redistribute the raw public data; download it from the DOI landing page and verify the files against `raw_file_hashes/sha256.json`.

## Frozen protocol

HUST pretraining is followed by supervised target-domain updating on the first 70% of each SCA test group in chronological order. The final 30% is an untouched time-out holdout. Case 1 post-replacement train data are used only as healthy baseline data. Thresholds are selected from the adaptation segment. Five seeds (11, 22, 33, 44, 55) are retained. `results/predictions_seed_*.json` contains probabilities, labels, timestamps, groups, and binary predictions for every held-out record.

## Reported result

Mean Macro-F1 = 0.2960 (SD 0.0115), Jaccard = 0.7198, exact set match = 0.7198, healthy-record false-alarm rate = 0.0062, fault-record recall = 0.2526, and mean warning lead for the two cases with explicit replacement dates = 12.6 days. The result supports only conditional utility after target adaptation; it does not establish zero-shot or universal field effectiveness.

## Archive step before submission

Upload this directory or its ZIP archive to a public Git repository or DOI-bearing archive, preserve the resulting persistent URL/version, and add that URL to the manuscript Data Availability statement.
