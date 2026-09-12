
## Cross-case holdout

`code/run_sca_leave_one_case_out.py` trains on two cases and evaluates the third with no records from the held case used for fitting or threshold selection. The held-case Macro-F1 ranges from 0 to 0.075 for DirectMLP and from 0 to 0.046 for the candidate, with unstable Jaccard values. This negative result is retained as evidence that same-case adaptation must not be described as cross-device generalization.
