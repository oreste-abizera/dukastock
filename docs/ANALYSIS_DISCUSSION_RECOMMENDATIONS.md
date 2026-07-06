# Analysis, Discussion & Recommendations

**Before filling this in**: the rubric describes all three of these
sections as done **"with the supervisor"** (Hubert Apana) — this reads
like it may be assessed through a live conversation/defense rather than a
written artifact submitted via Canvas. **Confirm this with your
supervisor or the course instructions before spending time writing a full
version of this document** — if it's purely conversational, this file
only needs to exist as a placeholder; if it does need to be submitted,
fill in the sections below after that conversation.

---

## Analysis

TODO — detailed analysis of how results were achieved, or how they missed
the proposal's objectives, with explicit linkage back to project scope.
Structure suggestion, one subsection per research question:

### RQ1 — Cold-start density thresholds (forecasting)
- What did the density × model benchmark actually find? (per-product
  winning model, threshold density at which each model class first beats
  naive — see `ml_experiments/results/ml_benchmark_results.csv`)
- Where did results match expectations from the literature (Fatima &
  Salam 2025; the 2025 XGBoost-on-sparse-data comparative study) vs.
  where they diverged?
- Honest scope note: evaluated on a Rwanda-localized **proxy** dataset
  (Kaggle Store Item Demand Forecasting Challenge), not real Duka
  transaction data — see `docs/RESEARCH_DESIGN.md`'s documented
  deviations section.

### RQ2 — XLM-R vs. RapidFuzz NER
- TODO once real annotated data exists: real precision/recall/F1/Kappa
  numbers, and what they say about MasakhaNER's Kinyarwanda-news backbone
  transferring to commerce-domain text.
- Until then: today's XLM-R fine-tuning run used the **synthetic
  placeholder** dataset — proves the training → serving → WhatsApp
  pipeline works end-to-end, but the resulting metrics are not a valid
  RQ2 result (see `RUNBOOK.md` Section 12).

### RQ3 — SUS usability
- TODO after the field session (`docs/SUS_QUESTIONNAIRE.md`) — scores by
  channel (WhatsApp vs. USSD), and what they suggest about the
  inclusive-access design goal (USSD for the 66% of households without a
  smartphone, per EICV7).

---

## Discussion

TODO — why the milestones reached this session matter, and what impact
the results have. Prompts to build from:

- The personalized-forecasting feature reuses the exact same
  walk-forward + Diebold-Mariano methodology from the offline research
  experiment, applied live per shopkeeper — a direct operationalization
  of the RQ1 finding into the serving layer, not just a research result
  sitting in a notebook.
- The system is deployed and demonstrably working end-to-end on a real
  device over a real messaging platform (WhatsApp), not just locally —
  what does that prove about feasibility for the target population
  (informal micro-retailers with only feature phones/USSD, or
  smartphone+WhatsApp)?
- What's the significance of the naive-baseline-until-proven-otherwise
  design (both offline and in the live personalized path) for a
  population with inherently short transaction histories?

---

## Recommendations

TODO — recommendations for the community/future work. Starting points:

- **For DukaStock's own future work**: real annotated NER data collection
  is the most impactful next step (unlocks a valid RQ2 result); the
  WelTel Rwanda SMS corpus (identified but access-gated during this
  session's research pass) is worth pursuing as a research partnership
  for a much larger real-world Kinyarwanda commerce-register dataset.
- **For similar low-resource-language NLP projects**: the gap between
  formal-register training data (news corpora like MasakhaNER) and
  informal target-domain text (commerce chat) is real and likely
  generalizes beyond Kinyarwanda — worth flagging as a broader lesson.
- **For deployment of similar systems**: Coolify (self-hosted PaaS) was a
  practical, low-cost path to a real production deployment with
  persistent storage for large ML artifacts, without cloud vendor
  lock-in — worth naming as a viable option for other resource-constrained
  student/research projects.
