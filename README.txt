Copy python/coef_audit.py next to compare_methods.py (python/ folder of the repo), then from python/:
python coef_audit.py --csv ../data/processed/X_losses.csv --start 1989-06-01 --end 1998-06-15 --boot 100 --win-boot 30 --topk 15
(about 5-8 minutes). Outputs in ../results/coef_audit/: 6 figures + boot_edge_coef.csv, rolling_features.csv, rolling_edge_coef.csv
