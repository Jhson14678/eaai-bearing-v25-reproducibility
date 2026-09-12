
## HUST strong baseline and orthogonal ablation

`code/run_hust_rf_baseline.py` adds a multi-output Random Forest baseline under the same development-only group cross-fitted threshold protocol. It obtains Macro-F1 = 0.5018 (SD 0.0124), Jaccard = 0.3595 and exact set match = 0 on the compound scoring vault.

`code/run_hust_orthogonal_ablation.py` changes one mechanism at a time under a fixed 35-epoch/0.002 learning-rate protocol (three seeds). Full, without-reconstruction, without-union, without-coherence, one-decoder-step and without-alignment variants are reported in `results/hust_orthogonal_ablation/results.json`.
