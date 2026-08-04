# Analysis, Discussion & Recommendations

## Analysis

### RQ1 — XLM-R vs. RapidFuzz NER (primary)

Two hundred real messages were collected directly from Duka shopkeepers
(`ml_experiments/data/Kinyarwanda Shopkeeper Dataset.xlsx`, plain-text
extract at `ml_experiments/data/shopkeeper_messages_for_doccano.txt`).
Entity-span annotation (PRODUCT/QUANTITY/UNIT) is complete for 183 of 200
messages (91.5%) per `docs/ANNOTATION_GUIDE.md`. XLM-R was fine-tuned on
the real annotated set (146 train / 37 eval, 15 epochs, CPU-only —
MPS ran out of memory during the optimizer step on 278M parameters) and
scored against RapidFuzz:

| Model | Precision | Recall | F1 |
|---|---|---|---|
| XLM-R (fine-tuned) | 0.953 | 0.965 | 0.959 |
| RapidFuzz (baseline) | 1.000 | 0.378 | 0.549 |

Cohen's Kappa on a 50-message subset independently labeled by two
annotators: 0.951 ("almost perfect," Landis & Koch, 1977). RapidFuzz's
recall failure is not a tuning artifact — it is architectural: it returns
at most one product per message and has no mechanism to disambiguate
which numeral in a sentence is the quantity versus, e.g., a price. XLM-R
correctly extracts two separate product/quantity/unit triples from a
single code-switched sentence ("Nabagurishije isukari ibiro bitatu
namavuta litre imwe" → SUGAR 3 kg and OIL 1 litre), something RapidFuzz
cannot do by design. The 37-message eval set is small; treat the exact
F1 as a point estimate on this domain and corpus, not a claim that
generalizes to arbitrary Kinyarwanda commerce text without further
validation.

### RQ2 — Cold-start density thresholds (forecasting; dropped from the final report's scope)

**Note (2026-07-19): this research question was removed from the final
capstone report once RQ1 reached a complete real result.** The analysis
below remains accurate to the code and is kept here as engineering
documentation, but it is not presented or defended in the thesis.



The density × model benchmark (`ml_experiments/results/ml_benchmark_results.csv`,
walk-forward cross-validation with a Newey-West-corrected Diebold-Mariano
test against the naive baseline, evaluated per Duka-proxy store and
averaged across all 10) shows a consistent pattern at full (100%) data
density: XGBoost achieves the lowest RMSE for every one of the five FMCG
products, but only clears the pre-registered significance bar
(`fraction_folds_significant >= 0.5`) for two of them — SUGAR (RMSE 4.74
vs. naive's 6.73, significant on 90% of folds) and SOAP (RMSE 8.90 vs.
13.40, significant on 60% of folds). For OIL, FLOUR, and RICE, XGBoost's
lower RMSE (8.81, 11.48, and 9.46 respectively, against naive's 12.12,
16.06, and 13.49) never reaches significance on more than 30% of folds,
so the naive baseline is correctly served instead — a lower point-estimate
error is not, by itself, evidence the model is reliably better than "same
weekday, last week."

This is the core finding operationalized directly into the live system:
`ForecastService`'s global artifacts and the per-shopkeeper personalized
service (`app/services/personalized_forecast_service.py`) both apply the
identical significance-gated selection rule, so the deployed product
never overstates confidence in a model that hasn't earned it. SARIMA and
Prophet were evaluated across the same matrix but never won on any
product at full density, and N-BEATS came closest of the three
non-XGBoost model classes without ever clearing the significance bar
either — consistent with XGBoost's documented strength on sparse,
intermittent retail demand (Fatima & Salam, 2025; arXiv:2506.05941) being
the more relevant regime here than SARIMA/Prophet's classical seasonal
decomposition strengths.

The dataset itself remains a Rwanda-localized **proxy** (the Kaggle Store
Item Demand Forecasting Challenge, mapped onto five Rwandan FMCG staples
with calendar/holiday features), not real Duka transaction data — this
limitation is disclosed in full in `docs/RESEARCH_DESIGN.md` and was a
deliberate, reasoned scope decision given the impracticality of
collecting multi-year real transaction histories from informal retailers
within the project timeline, not an oversight.

### RQ3 — SUS usability

The SUS field session with real Duka operators has not yet been
conducted as of this submission, so no usability score data exists yet
to analyze. The questionnaire (English and Kinyarwanda), consent form,
and administration protocol are all prepared and ready
(`docs/SUS_QUESTIONNAIRE.md`, `docs/CONSENT_FORM.md`).

---

## Discussion

The most significant outcome of this milestone is that the offline
research finding (RQ2's significance-gated model selection) is no longer
confined to a notebook — it's the exact rule the live, deployed system
uses to decide what to tell a real shopkeeper, at both the global and the
per-shopkeeper personalized level. That distinction matters because it
closes the usual gap between "a model that scored well in an experiment"
and "a model a production system actually trusts enough to serve,"
without any additional translation step or re-derivation between the two.

The system is also demonstrably working end-to-end on a real device over
a real messaging platform, not only in local development. That matters
directly for the target population: the same backend serves USSD (via
Africa's Talking, reachable from any phone including the feature phones
the majority of surveyed households in Rwanda's EICV7 report using) and
WhatsApp (for shopkeepers with a smartphone), so the inclusive-access
design goal is not just a diagram in the architecture document but a
system that has actually processed a real message from a real phone.

Finally, the naive-baseline-until-proven-otherwise design — applied
identically offline and in the live personalized-forecasting path — is a
direct, practical answer to the reality that most real shopkeepers using
this system for the first time will have very little transaction
history. Rather than serving an unproven model's confident-looking number
to a brand-new user, the system is honest about what it doesn't yet know,
which is a defensible product decision as much as a research one.

---

## Recommendations

For DukaStock's own continuation: completing real NER annotation is the
single highest-leverage remaining step, since the training pipeline,
serving path, and WhatsApp integration are already proven — only the
underlying data is outstanding. Separately, the WelTel Rwanda SMS corpus
identified during this project's data-sourcing research (real two-way
Kinyarwanda health-system SMS, currently access-gated under a research
partnership) is worth pursuing directly with that research team as a
much larger real-world commerce-adjacent-register dataset than any
single student project could collect independently.

For similar low-resource-language NLP projects more broadly: the gap
between formal-register training data (existing Kinyarwanda NLP
resources like MasakhaNER are sourced from news text) and informal,
domain-specific target text (commerce chat, in this case) is real and
consistently underestimated — teams building for a specific informal
register should budget real time and effort for primary data collection
rather than assuming an adjacent public dataset will transfer.

For deployment of similar resource-constrained research systems: Coolify,
a self-hosted PaaS running on a single VPS, proved to be a practical path
to a real production deployment — including persistent storage for
large ML model artifacts and scheduled batch jobs — without the ongoing
cost or vendor lock-in of a managed cloud platform, and is worth
recommending to other student or early-stage research projects facing
the same budget constraints.
