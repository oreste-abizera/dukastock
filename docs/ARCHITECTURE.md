# DukaStock Architecture

See `docs/diagrams/figure2_architecture.svg` (or `.png`) for the rendered
version of the diagram below — the ASCII art here is a quick-reference
copy for viewing directly in a terminal or plain-text editor.

## Overview

A single FastAPI backend serves three channels through a shared NLP and
forecasting pipeline:

```
                    ┌──────────────────┐
   WhatsApp  ──────▶│                  │
   (Twilio)         │                  │
                     │                  │      ┌─────────────┐
   USSD      ──────▶│   FastAPI        │─────▶│  Forecasting │
   (Africa's        │   Backend        │      │   Service    │
   Talking)          │                  │      └──────┬──────┘
                     │                  │             │
   SMS (out) ◀──────│                  │      ┌──────▼──────┐
   (Africa's         └─────────┬────────┘      │  joblib      │
   Talking)                    │               │  artifacts   │
                                │               └─────────────┘
                     ┌──────────▼────────┐
                     │  Supabase (Postgres)│
                     │  + Redis (Upstash)  │
                     └────────────────────┘
```

## Why this stack

- **FastAPI**: handles two concurrent inbound webhook providers (Twilio +
  Africa's Talking). Webhook route handlers are plain `def`, not
  `async def`, by design — every downstream call (SQLAlchemy's sync
  `Session`, the NER pipeline, SARIMA/Prophet/XGBoost inference) is
  itself blocking/CPU-bound, so FastAPI dispatches each request to its
  threadpool automatically. Declaring these routes `async def` without
  first moving the DB layer to an async driver would block the single
  event loop on every request instead of just one worker thread — a
  regression, not an improvement. A true async stack (asyncpg +
  SQLAlchemy `AsyncSession` end to end) is the correct next step if
  webhook volume ever requires it.
- **Railway.app over Render.com**: Render's free tier cold-starts after
  inactivity, which would breach the ~3-second USSD response SLA that MTN
  and Airtel enforce at the telecom layer. Railway's always-on tier avoids
  this. Build/deploy config lives in `railway.json` (Dockerfile-based
  build, health check at `/api/v1/health`); connecting the Railway
  service to this repo and triggering the actual deploy is a manual,
  one-time dashboard step — GitHub Actions runs CI (lint, tests, Docker
  build) but does not push a deploy on its own.
- **Redis with a 180-second TTL**: USSD is stateless per-request at the
  telecom layer — every keypress is a brand new HTTP call to our webhook
  with the full accumulated input string. Redis is the FSM's memory across
  those calls, and the TTL is set to match the network's own session
  timeout so state never outlives the session it belongs to.
- **Phone numbers are never stored raw**: every inbound number is hashed
  (`app.core.security.hash_phone_number`) before it reaches the database
  layer, in compliance with Rwanda's Law No. 058/2021.

## Channel-specific behaviour

| Channel | Input parsing | Output constraint |
|---|---|---|
| WhatsApp | Free text → XLM-R NER (RapidFuzz fallback) | None (Twilio handles segmentation) |
| USSD | Structured FSM menu, no NLP | 182 characters (Africa's Talking USSD limit) |
| SMS | N/A (outbound only) | 160 characters (GSM-7 single segment) |

## Known correction from the proposal draft

The proposal draft's Table 7 (Development Tools) and Chapter 3.3 originally
stated Rwanda has "12 public holidays." Independent verification during
this build (see `docs/SOURCES.md`) confirms the correct figure is **14**.
The codebase (`app/ml/pipeline/rwanda_features.py`) implements all 14
correctly; the final written proposal document should be updated to match.

## Data flow for the secondary research question (forecasting benchmark)

1. `ml_experiments/data/train.csv` (Kaggle Store Item Demand Forecasting
   Challenge, 913K rows) is loaded by Notebook 1, which subsets it to 5
   Rwanda-mapped FMCG products (`subset_fmcg_products`) and attaches Rwanda
   features (`add_rwanda_features`), producing `fmcg_rwanda_localized.csv`.
2. Notebook 2 / `run_experiment.py` load `fmcg_rwanda_localized.csv` and
   iterate over each of the 10 stores per product. Each (store, product)
   pair is treated as a **Duka-proxy series** — the unit of analysis that
   matches the deployment context (one shopkeeper, one shop).
3. For each (store, product) pair, six temporal cold-start slices are
   produced (`temporal_density_slice`) at 5%, 15%, 30%, 50%, 75%, 100%
   of the chronological series, simulating a Duka with differing amounts
   of recorded history.
4. Within each slice, walk-forward folds are generated (`walk_forward_folds`,
   minimum 6 folds, 7-day horizon) so every train/test split respects
   chronological order.
5. Five model classes are fit and scored per fold: Naive, SARIMA, Prophet,
   XGBoost (TimeSeriesSplit grid search), N-BEATS (500 steps on GPU).
6. The Diebold-Mariano test (Newey-West HAC variance, h=7) compares each
   model's squared-loss sequence against the naive baseline's, fold by fold.
   `significant_at_05` is `True` only when both p < 0.05 AND d_bar < 0
   (model is directionally better, not just statistically different).
7. Metrics are averaged across the 10 stores. The **threshold density** —
   the lowest density where a model achieves DM significance on ≥ 50% of
   folds (averaged across stores) — is the headline finding per model class.
8. The best model at 100% density (lowest mean RMSE, store-averaged) is
   serialized as `artifacts/<product>_best_model.joblib` for the API.

This same logic is implemented twice, intentionally: once as a CI-friendly,
headless script (`ml_experiments/scripts/run_experiment.py`) and once as an
exploratory, heavily visualized notebook
(`ml_experiments/notebooks/02_ml_benchmark_experiment.ipynb`). Both call
into the identical `app.ml.*` modules; results from either entry point are
directly comparable.
