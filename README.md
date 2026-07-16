# DukaStock

Teaching a language model to understand how Rwandan Duka shopkeepers
actually text about their sales — with demand forecasting and a
three-channel delivery system (WhatsApp, USSD, SMS) built around it.

Capstone project — BSc. Software Engineering (Machine Learning), African
Leadership University, Kigali. Supervisor: Hubert Apana.

## Deployed version

**Live at: https://api.dukastock.oreste.dev** — health check:
`GET /api/v1/health`, interactive API docs: `/docs`. Deployed on a
self-hosted VPS via Coolify; see `RUNBOOK.md` Section 13 for the full
deployment writeup.

## Demo video

**Watch: https://www.bugufi.link/bWofKw** — a walkthrough of core
functionality: WhatsApp sale logging via the fine-tuned NER model, the
USSD menu including sales history and language switching, and a
forecast lookup.

## Testing results & analysis

- [`docs/TESTING_RESULTS.md`](docs/TESTING_RESULTS.md) — testing evidence
  across different strategies, data values, and hardware/software
  environments.
- [`docs/ANALYSIS_DISCUSSION_RECOMMENDATIONS.md`](docs/ANALYSIS_DISCUSSION_RECOMMENDATIONS.md)
  — analysis, discussion, and recommendations.

**New to this repo? Start with [`RUNBOOK.md`](RUNBOOK.md)** — a complete,
numbered walkthrough from an empty machine to a running system, including
the test suite, the notebooks, real data collection, and model training.
Everything below is reference material; the runbook is the thing to
actually follow step by step.

## What this is

Informal kiosk shops ("Dukas") make up the majority of Rwanda's retail
sector but keep zero digital sales records, so every published ML demand
forecasting method — which assumes historical transaction data — is
structurally unusable for them. This project:

1. **Evaluates a fine-tuned XLM-R model** on a 200-message annotated
   commerce-domain Kinyarwanda-English NER test set, comparing it against
   a RapidFuzz rule-based baseline — the primary research question,
   since natural-language understanding on the WhatsApp channel is the
   project's most immediately legible contribution.
2. **Characterizes the minimum transaction history** at which four ML
   model classes (SARIMA, Prophet, XGBoost, N-BEATS) first beat a naive
   baseline with statistical significance (Diebold-Mariano test, Newey-West
   HAC variance, p < 0.05), evaluated at the individual-store (Duka-proxy)
   level across six cold-start data density levels, using a Rwanda-localized
   version of the Kaggle Store Item Demand Forecasting Challenge benchmark.
3. **Delivers a working three-channel prototype** (WhatsApp via Twilio,
   USSD via Africa's Talking for MTN/Airtel Rwanda, outbound SMS) so the
   research has a usable interface, not just a results table.

## Repository layout

```
dukastock/
├── backend/              FastAPI application (see backend/README implicit in app/ structure)
│   └── app/
│       ├── core/          config, security (phone hashing), logging
│       ├── db/            Supabase session + Redis client
│       ├── models/        SQLAlchemy ORM (matches proposal class diagram exactly)
│       ├── ml/
│       │   ├── pipeline/   Rwanda feature engineering + cold-start splitting
│       │   ├── evaluation/ RMSE/MAE/MAPE/sMAPE + Diebold-Mariano test
│       │   └── models/     naive, SARIMA, Prophet, XGBoost, N-BEATS
│       ├── nlp/            XLM-R + RapidFuzz commerce NER pipeline
│       ├── channels/       whatsapp/, ussd/ (FSM), sms/
│       └── api/v1/         router (5 endpoints)
│   └── tests/unit/        97 tests covering metrics, security, features, models, NLP, FSM, personalized forecasting
├── ml_experiments/
│   ├── scripts/            run_experiment.py (CLI benchmark), train_xlmr_ner.py
│   ├── notebooks/          4 Kaggle-ready Jupyter notebooks (see below)
│   ├── data/                place Kaggle train.csv here
│   └── results/             experiment JSON output lands here
├── docs/                   ARCHITECTURE.md, RESEARCH_DESIGN.md, SOURCES.md, ANNOTATION_GUIDE.md, SUS_QUESTIONNAIRE.md
│   ├── TESTING_RESULTS.md   Testing evidence (strategies, data values, hardware/software)
│   ├── ANALYSIS_DISCUSSION_RECOMMENDATIONS.md
│   ├── screenshots/          Screenshot evidence referenced from TESTING_RESULTS.md
│   └── diagrams/            Figures 2, 4, 5, 6, 7 (architecture, use case, class, ER, sequence) as SVG + PNG
├── .github/workflows/ci.yml
├── docker-compose.yml
└── config/.env.example
```

## Notebooks

| # | Notebook | Contents |
|---|---|---|
| 1 | `01_eda_rwanda_localisation.ipynb` | Explores the raw Kaggle dataset, applies and visualizes the Rwanda holiday/season localisation layer |
| 2 | `02_ml_benchmark_experiment.ipynb` | Full density × model benchmark, walk-forward CV, Diebold-Mariano testing, threshold curve |
| 3 | `03_xlmr_commerce_ner.ipynb` | XLM-R fine-tuning on the commerce NER test set, RapidFuzz comparison, Cohen's Kappa |
| 4 | `04_results_dashboard.ipynb` | Publication-quality figures synthesizing notebooks 1–3 for the thesis defense |

Each notebook is self-contained and Kaggle-runnable (installs its own
dependencies in the first cell, expects `train.csv` either uploaded as a
Kaggle dataset or placed in `ml_experiments/data/`).

## Quick start (local development)

For the full walkthrough with verification steps and troubleshooting, see
[`RUNBOOK.md`](RUNBOOK.md). Condensed version:

```bash
git clone <this-repo>
cd <the-cloned-folder>   # named per your local clone/unzip, not necessarily "dukastock"
cp config/.env.example config/.env   # fill in Twilio/Africa's Talking sandbox keys
docker-compose up --build
```

The API will be available at `http://localhost:8000`. Health check:
`GET /api/v1/health`.

### Running tests

```bash
cd backend
pip install -r requirements.txt
PYTHONPATH=. DATABASE_URL=sqlite:///./test.db pytest tests/unit/ -v
```

### Running the ML benchmark

```bash
# 1. Download train.csv from
#    https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data
#    into ml_experiments/data/, then run Notebook 1 to produce the
#    Rwanda-localized dataset (fmcg_rwanda_localized.csv).
#    Alternatively, run the script directly on train.csv from Notebook 1's output.

python ml_experiments/scripts/run_experiment.py \
  --data ml_experiments/data/fmcg_rwanda_localized.csv \
  --output-dir ml_experiments/results \
  --artifact-dir ml_experiments/artifacts
```

Evaluation is at the **individual-store (Duka-proxy) level** — each of the
10 Kaggle stores is treated as a proxy for one Duka. Results are averaged
across stores and saved as `ml_experiments/results/ml_benchmark_results.csv`.
**Runtime: ~10–14 hours on Kaggle/Colab GPU** (SARIMA order-search is the
bottleneck). Use Kaggle's 12h GPU sessions or Colab T4.

### Running the XLM-R NER fine-tuning

```bash
# Collect and annotate 200 real Kinyarwanda commerce messages first —
# see docs/ANNOTATION_GUIDE.md for tooling and the second-annotator
# protocol used to compute Cohen's Kappa. Without this, the notebook
# and script fall back to synthetic placeholder data (clearly flagged
# in their output) for pipeline validation only.
python ml_experiments/scripts/train_xlmr_ner.py \
  --annotations ml_experiments/data/annotations.jsonl \
  --output-dir backend/app/nlp/xlmr_commerce_ner
```

## Privacy

Phone numbers are never stored in plaintext. Every inbound number is
hashed (SHA-256 + UUID5, salted) at the point of ingestion, per Rwanda's
Law No. 058/2021 relating to the protection of personal data and privacy.
See `app/core/security.py` and `docs/SOURCES.md`.

## Known correction from the proposal draft

The written proposal's Table 7 and Chapter 3.3 state Rwanda has "12 public
holidays." Independent verification (see `docs/SOURCES.md`) found the
correct figure is **14**. The codebase implements all 14 correctly;
`docs/ARCHITECTURE.md` flags this for correction in the final written
document.
