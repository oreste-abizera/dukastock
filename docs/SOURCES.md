# Verified Sources

Every Rwanda-specific statistic and external technical claim used in this
codebase and in the accompanying notebooks was checked against a live web
source during this build (June 2026), independent of the proposal draft.
This file exists so the claims in `docs/RESEARCH_DESIGN.md`, the notebooks,
and the proposal can all be traced back to primary evidence.

## Rwanda economic / retail context

| Claim | Verified value | Source |
|---|---|---|
| Wholesale & retail trade share of all establishments | 53.5% (2023) | NISR Establishment Census 2023, statistics.gov.rw |
| Informal share of Rwanda's business landscape | Confirmed via NISR Establishment Census 2023 (92.0% sole proprietorship, 92.2% micro-enterprises) and Integrated Business Enterprise Survey 2024 | statistics.gov.rw |
| Households owning a smartphone | 34% (2024) | NISR EICV7, reported via allAfrica.com (16 Apr 2026) |
| Overall mobile phone ownership | 85% (2024), up from 67% (2017) | NISR EICV7 |
| Mobile internet usage | 20% nationally | EICV7, via TechCabal (12 Jun 2025) |
| Urban vs. rural internet usage | 57% urban / 19% rural | EICV7 |

## Rwanda public holidays

| Claim | Verified value | Source |
|---|---|---|
| Number of official public holidays | **14**, not 12 | Wikipedia "Public holidays in Rwanda", cross-checked against officeholidays.com and calendarific.com (23 holiday *instances* in 2026 because Eid dates + weekend-shift compensation days are counted separately by some trackers, but the 14 *categories* figure is the one that matches "official Rwandan public holidays" as a count of distinct observances) |
| Genocide Memorial Day exception | 7 April is never shifted to a working day even if it falls on a weekend; the following week is an official week of mourning | Wikipedia "Public holidays in Rwanda" |

This codebase uses **14** as the holiday category count (`rwanda_features.py`), correcting an earlier draft figure of 12 found in a prior version of the proposal.

## Rwanda data protection law

| Claim | Verified value | Source |
|---|---|---|
| Law number and date | Law N° 058/2021 of 13/10/2021, gazetted 15 October 2021 | risa.gov.rw, dpo.gov.rw, RwandaLII |
| Supervisory authority | National Cyber Security Authority (NCSA) | dpo.gov.rw |

## Comparable platforms

| Claim | Verified value | Source |
|---|---|---|
| Sauti East Africa user count | 150,000+ farmers/traders **across Kenya, Rwanda, Tanzania and Uganda combined** — not Rwanda alone | Centre for Humanitarian Data (humdata.org) |
| Baseline trader information cost before Sauti | USD 3.50–13 per week | sautiafrica.org case study |
| Africa's Talking channel support in Rwanda | SMS (bulk & short code) and USSD only — **no WhatsApp product** in Rwanda via Africa's Talking | help.africastalking.com |

This confirms the proposal's architecture decision to source WhatsApp via Twilio specifically (not Africa's Talking) is technically necessary, not arbitrary.

## ML / NLP benchmarks

| Claim | Verified value | Source |
|---|---|---|
| Kaggle Store Item Demand Forecasting Challenge size | 913,000 rows; 10 stores × 50 items; daily; 2013–2017 | Kaggle competition page; cross-checked against 3 independent reproductions (GitHub, Medium) |
| MasakhaNER 2.0 XLM-R-base average F1 | 84.1 ± 0.1 | ACL Anthology 2022.emnlp-main.298 |
| N-BEATS M4 improvement | Oreshkin et al., ICLR 2020 | arxiv.org/abs/1905.10437 |

## Note on this document's scope

This file documents facts checked during the **rebuild** described in this
conversation. It deliberately does not carry over or assume any fact-check
performed in earlier, separate sessions — every figure above was
independently re-verified.

## Replacing synthetic placeholder data

Notebook 3 (`03_xlmr_commerce_ner.ipynb`) generates synthetic placeholder
commerce messages when no real annotation file is present at
`ml_experiments/data/annotations.jsonl`, purely so the fine-tuning and
evaluation pipeline can be validated end-to-end without real data. See
`docs/ANNOTATION_GUIDE.md` for how to collect and annotate a real 200-message
test set (tooling, format, and the second-annotator protocol for Cohen's
Kappa) before reporting any NER numbers in the thesis.
