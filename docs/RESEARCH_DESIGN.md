# Research Design

Mirrors proposal Chapter 3.2 ("Research Design") and Chapter 1.4 ("Research
Questions"). This document exists so the experimental protocol is
discoverable directly from the codebase, not only from the Word document.

## Primary research question

How accurately does a fine-tuned XLM-R model extract product names and
quantities from informal English-Kinyarwanda code-switched commerce
messages compared to a rule-based RapidFuzz baseline, measured by
precision, recall, and F1 on a 200-message annotated test set with
reported Cohen's Kappa?

**Status as of this report**: data collection is complete (200 real
messages collected directly from Duka shopkeepers). Entity-span
annotation in Doccano is in progress (26/200, 13%, as of 2026-07-06). The
full precision/recall/F1/Cohen's Kappa result on the real annotated set is
pending completion of annotation; this section will be updated once
available. The pipeline itself (fine-tuning, serialization, serving, live
WhatsApp delivery) is already verified working end-to-end, on the
synthetic placeholder set, ahead of the real result.

## Secondary research question

Across SARIMA, Prophet, XGBoost, and N-BEATS, at what minimum data density
does each model class first achieve statistically significant improvement
over a naive last-week-sales baseline (p < 0.05, Diebold-Mariano test with
Newey-West HAC variance correction for h = 7), evaluated at the individual
store (single-Duka proxy) level, when trained under simulated cold-start
conditions on a Rwanda-localized formal retail benchmark dataset?

## Tertiary research question

What usability patterns emerge from System Usability Scale scores when
Duka operators interact with the WhatsApp and USSD prototype channels?

## Why temporal cold-start, not random masking

Bergmeir & Benitez (2012) established that standard k-fold cross-validation
is invalid for time series because it lets information from the future
leak into training. Suradhaniwar et al. (2021) extended this finding
specifically to SARIMA and ML demand models. DukaStock's six density
levels (5/15/30/50/75/100%) are therefore **temporal prefixes** of the
chronological series, not randomly sampled subsets — `temporal_density_slice`
in `app/ml/pipeline/cold_start.py` always takes the *first* N% of dates,
because that is what "a Duka that started recording sales recently" 
actually looks like.

## Why walk-forward validation, not a single train/test split

A single split would only test one moment in time. Walk-forward validation
(`walk_forward_folds`) re-trains on an expanding window and tests on the
next 7 days, repeated at least 6 times per density level, so the threshold
density finding is robust to which particular week happened to be the test
week.

## Why the Diebold-Mariano test, not just comparing RMSE numbers

Two models can have different RMSE purely by chance, especially on short,
sparse cold-start series. The DM test (Diebold & Mariano, 1995) asks
whether the *difference* in loss is large enough, relative to its own
variance across folds, to rule out chance. This is what makes "minimum
data density" a defensible empirical finding rather than an eyeballed
number from a chart.

## Why these five model classes specifically

| Model | Role | Why included |
|---|---|---|
| Naive (last-week-sales) | H0 / performance floor | Always available with zero training; what every Duka owner does mentally already |
| SARIMA | Classical statistical baseline | Falatouri et al. (2022): competitive on short series with clear seasonal structure |
| Facebook Prophet | Calendar-aware decomposable baseline | Accepts the Rwanda holiday calendar directly; Taylor & Letham (2018) |
| XGBoost + Rwanda features | Primary ML candidate | Fatima & Salam (2025) and arXiv:2506.05941 found tree ensembles strongest under sparse intermittent demand — the exact regime of a cold-start Duka |
| N-BEATS | Lightweight neural baseline | Oreshkin et al. (2020): +11% over the M4 statistical benchmark with zero feature engineering, useful as a check on whether hand-engineered Rwanda features are even necessary |

## Why these five products

Sugar (isukari), cooking oil (amavuta yo guteka), flour (ifu), rice
(umuceri), and soap (isabune) were chosen as Rwandan FMCG staples with
genuine demand variability. Airtime was explicitly excluded in the
proposal scope because its demand is near-constant and offers no
forecasting challenge — there would be nothing for any model, including
the naive baseline, to get wrong.

## Reproducibility

All code, configurations, and evaluation scripts are designed to run
identically whether invoked from `ml_experiments/scripts/run_experiment.py`
(headless, CI-friendly) or from the notebooks in `ml_experiments/notebooks/`
(exploratory, fully visualized) — both call into the same `app.ml.*`
modules in `backend/`.

## Honest limitations — disclosed in thesis methods section

### Dataset is a proxy, not real Rwandan retail data

The benchmark dataset is the **Kaggle Store Item Demand Forecasting
Challenge** (10 anonymous stores, 50 anonymous items, 2013-2017). This
dataset was chosen because no publicly available time-series dataset of
informal Rwandan Duka sales exists (see `docs/SOURCES.md`). It is used as a
structural proxy to test whether cold-start density affects model accuracy
under weekly-seasonal demand. All Rwanda-specific features (holiday calendar,
Genocide Memorial Day suppressor, rainy-season intensity, FMCG product naming)
are **fully implemented and designed for deployment** on real Duka sales data
but cannot be empirically validated on this proxy dataset.

Downstream claims made in this thesis are restricted to:
- "Under a proxy dataset structurally similar to weekly FMCG retail demand..."
- "The Rwanda localisation layer is production-ready for real Duka data..."
- NOT: "Memorial Day reduces sales by X%" or "Rain increases demand by Y%"
  — these claims cannot be substantiated and are not made.

### Per-store evaluation rationale

Evaluation is performed at the individual store (Duka-proxy) level, not as
a 10-store national aggregate. Rationale: DukaStock is deployed to individual
shopkeepers. Aggregating 10 stores into one national series inflates lag-7
autocorrelation from r = 0.55-0.80 (per-store) to r = 0.94 (national),
leaving only 12% of variance for any model to explain and making
Diebold-Mariano significance nearly unachievable regardless of model quality.
The per-store evaluation gives a fair test that matches the actual deployment
unit. Each of the 10 Kaggle stores serves as an independent Duka proxy;
metrics are averaged across stores and reported with standard deviations.

### DM test statistical power disclosure

At 6 walk-forward folds per (store, density) combination with n = 7 test
observations per fold (total n = 42 loss-differential observations), the
DM test has limited statistical power. Effect sizes need to be practically
large to achieve p < 0.05. This is disclosed explicitly in the results
section: "Failure to achieve statistical significance at low density levels
is expected given n ≤ 42 loss-differential observations per DM test — it
reflects limited power, not model equivalence."

### XGBoost product-feature mapping is arbitrary on this dataset

The mapping {item_1: SUGAR, item_7: OIL, item_13: FLOUR, item_24: RICE,
item_35: SOAP} is implemented for deployment readiness. The Kaggle dataset
items are anonymous; the mapping cannot be verified empirically.

## Documented deviations from the written proposal

Two implementation choices depart from a literal reading of proposal
Chapter 3.3 ("Machine Learning Training Pipeline"). Both are deliberate,
reasoned decisions made during implementation, not oversights.

1. **Decided: Prophet does not use lag-1/2/4 week features as regressors.**
   The proposal's pipeline description groups lag features under "XGBoost
   and Prophet." This has been evaluated and the decision is final —
   lag features stay XGBoost-only. Reasoning, for the thesis methodology
   section (Chapter 3.3):

   > Prophet's own `weekly_seasonality` term already models the same
   > 7-day periodicity that a `lag_7d` regressor would encode. Supplying
   > it anyway would not give the model new information — it would let
   > Prophet re-derive, via an external regressor, a signal already
   > present in its seasonal decomposition, inflating its apparent fit
   > without testing anything additional. This would weaken rather than
   > strengthen the four-model comparison. Prophet's differentiating
   > property in this study (Table 5) is calendar-awareness — the
   > Rwanda public holiday and Genocide Memorial Day suppressor
   > injected via its holidays parameter — which is implemented in
   > full. Lag features are therefore scoped to XGBoost only, the model
   > class for which they are proposed as a primary strength.

   Drop that paragraph into Chapter 3.3 directly, adjusting only for
   voice/tense to match the rest of the chapter. See the module docstring
   in `backend/app/ml/models/prophet_model.py` for the original
   implementation-time reasoning this paragraph is drawn from.

2. **Rwanda's holiday count is implemented as 14, not the 12 stated in
   the proposal text.** Independent verification (see `docs/SOURCES.md`)
   found 14 official categories of public holiday. The written proposal
   document (Chapter 3.3, "Rwanda's official 12 public holidays") should
   be corrected to match the code, or the code should be deliberately
   reverted to 12 if there's a specific reason to match the original
   figure (e.g. a cited source the supervisor prefers) — this has not been
   done automatically in either direction, since it changes a stated fact
   in the proposal text itself.

3. **The NLP research question (originally secondary) is now designated
   primary, and the forecasting question (originally primary) is now
   secondary.** This reverses the ordering in the original approved
   proposal. Cleared with the supervisor (Hubert Apana) ahead of the final
   report. Reasoning for the thesis introduction/methodology section:

   > The WhatsApp channel's natural-language understanding is the
   > system's most immediately legible contribution — a shopkeeper
   > interacting with free-text Kinyarwanda/English commerce messages,
   > with no menu or form to learn, is the clearest illustration of why
   > this project's approach differs from existing demand-forecasting
   > tools. Elevating it to the primary research question better reflects
   > where the project's emphasis and novelty lie.

   Note for the written report: at the time of this reordering, the
   primary question's own real-data result (precision/recall/F1/Kappa on
   the real annotated set) is still in progress — see the "Status as of
   this report" note under the Primary research question above. The
   secondary (forecasting) question is the one with a complete, fully
   evaluated result already operationalized in the deployed system.
   Both facts should be stated plainly in the report rather than implied.
