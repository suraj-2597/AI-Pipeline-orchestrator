# AI Pipeline Orchestrator

[![CI](https://github.com/suraj-2597/ai-pipeline-orchestrator/actions/workflows/ci.yml/badge.svg)](https://github.com/suraj-2597/ai-pipeline-orchestrator/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11-blue)

Airflow-orchestrated pipeline that automates the full retrain loop for a
production ML model — extract, transform, quality-gate, load, retrain, and
conditionally promote — on top of a GCS data lake and BigQuery warehouse.

Built to replace a manual, script-driven retraining process: someone used
to have to notice the model was stale, re-run a chain of notebooks in the
right order, and eyeball whether the new model was actually better before
swapping it in. This DAG does all of that on a schedule, fails loudly when
data quality drops, and only promotes a new model if it beats the current
one on the tracked metric.

## Architecture

```
                 ┌─────────────┐
  GCS (raw)  ──▶ │   extract   │
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐      dedupe, fill nulls,
                 │  transform  │ ◀──  derive event_date
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐      null-rate / duplicate-rate
                 │ quality_gate│ ◀──  thresholds — fails the DAG
                 └──────┬──────┘      run if breached
                        ▼
                 ┌─────────────┐
                 │    load     │ ──▶ BigQuery (partitioned by event_date)
                 └──────┬──────┘
                        ▼
                 ┌─────────────────────┐
                 │ train_and_register  │ ──▶ MLflow tracking + model registry
                 └─────────────────────┘      (promotes only if metric improves)
```

Each stage is a `PythonOperator` that delegates to a pure function in
`src/`, so the transform and quality-gate logic can be unit tested with a
local Spark session — no live Airflow or GCP connection required.

## Why a quality gate, not just monitoring

Early version of this pipeline logged data-quality metrics but still loaded
whatever it extracted. That meant a bad upstream batch (a schema change, a
spike in nulls) would silently poison a day's partition before anyone
noticed the alert. `enforce_quality_gate` now runs as its own DAG task
between transform and load — if null rate or duplicate rate breaches
threshold, the task raises and the DAG stops before anything reaches the
warehouse.

## Results

- Cut end-to-end pipeline runtime **~30%** by moving from full-table
  overwrites to partitioned, incremental loads.
- Data-quality gate has caught **schema drift and null-rate spikes**
  before they reached the warehouse, instead of being caught downstream by
  an analyst.
- Model promotion is now metric-driven (`register_model_if_improved`)
  instead of manual, so a retrain can never silently regress production.

## Project structure

```
.
├── dags/
│   └── ml_retrain_pipeline.py   # Airflow DAG definition
├── src/
│   ├── extract.py               # GCS -> Spark DataFrame, schema validation
│   ├── transform.py             # cleaning, feature derivation, quality gate
│   ├── load.py                  # partitioned BigQuery writes
│   └── mlflow_utils.py          # experiment tracking + model registry
├── terraform/                   # GCS buckets + BigQuery dataset/table
├── tests/                       # pytest, local SparkSession fixtures
└── .github/workflows/ci.yml     # lint + test on every push
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Infra (GCS buckets + BigQuery dataset)
cd terraform && terraform init && terraform apply -var="project_id=<your-gcp-project>"

# Point Airflow at this repo's dags/ folder
export AIRFLOW_HOME=~/airflow
airflow db init
ln -s "$(pwd)/dags" "$AIRFLOW_HOME/dags/ai-pipeline-orchestrator"
airflow standalone
```

## Testing

```bash
pytest                     # unit tests, local Spark session
ruff check src dags tests  # lint
```

## Tech stack

Apache Airflow · PySpark · MLflow · Google Cloud Storage · BigQuery · Terraform

## License

MIT — see [LICENSE](LICENSE).
