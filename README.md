# RMLM Industry Portfolios -- Implementation & Research Project

Implementation + reproduction + research-gap exploration for:
Klüppelberg, C. & Krali, M. (2025/2026), "Causal analysis of extreme risk in a
network of industry portfolios", arXiv:2504.00523 / Canadian J. Statistics.

See docs/ (in Persian) for the full walk-through: project structure, what each
file is for, how to run the data pipeline, how to test, and a list of
research-gap experiments with starter code.

Quick start:
    cd python
    pip install -r ../environment/requirements.txt
    pytest tests/ -v
    python run_pipeline.py        # needs data/raw/30_Industry_Portfolios_Daily.CSV
