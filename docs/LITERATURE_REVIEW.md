# Literature Review — Draft

**Before using this in the final report**: this is a first-pass draft to
adapt into your own voice, not text to submit as-is. Every citation below
was verified against a primary source during drafting (title, authors,
venue all confirmed), but the *specific claims* attributed to each paper
here are based on abstracts and search summaries, not a full read of each
paper. Read each source yourself before finalizing any sentence that
makes a specific claim about what a paper found or argued — this protects
you from misrepresenting a paper you haven't actually read closely.

Organized to match the report's primary/secondary research question
ordering: NLP (primary) gets the most developed sections; forecasting
(secondary) is covered with proportionally less depth, consistent with it
already being your most complete, fully-evaluated result.

---

## 1. Code-switching and low-resource language NLP

Kinyarwanda-English-French code-switching is not a marginal case in
Rwandan digital communication — it is close to the default register for
informal messaging, and any system built to parse real commerce messages
has to treat it as the norm rather than an edge case. Winata et al.
(2023) provide the most comprehensive recent synthesis of code-switching
research in NLP, tracing the field's progression from early rule-based
detection of language boundaries to the current generation of
transformer-based approaches, and — critically for this project —
cataloguing which NLP subtasks (sentiment analysis, machine translation,
named entity recognition among them) have received sustained attention
and which remain comparatively under-served. Commerce-domain entity
extraction sits closer to the latter category: the survey's coverage of
NER-related code-switching work is dominated by high-resource language
pairs (Hindi-English, Spanish-English) and general-domain text, leaving
domain-specific, low-resource-language commerce text — precisely this
project's setting — largely unaddressed.

This gap is not specific to code-switching research; it reflects a
broader pattern in low-resource language NLP. Magueresse, Carles, and
Heetderks (2020) frame the core structural problem: NLP progress has
historically been driven by the availability of large annotated corpora
and native-speaker expert time, both of which are scarce for languages
like Kinyarwanda relative to English, Mandarin, or Spanish. Their review
of "past work and future challenges" positions data scarcity not as an
implementation inconvenience but as the defining methodological
constraint that shapes what is achievable — directly motivating this
project's own decision to prioritize collecting and annotating a small,
real, domain-specific dataset (Section [X]) over attempting to source or
construct a larger but less representative one.

## 2. Transformer architectures and domain adaptation

The technical foundation for this project's primary contribution rests
on two developments in transfer learning for NLP. Devlin et al. (2019)
established the pretrain-then-fine-tune paradigm — a single deep
bidirectional transformer, pretrained on unlabelled text at scale, then
fine-tuned with a small task-specific head — that essentially all
subsequent work in this space, including this project's own pipeline,
still follows. Conneau et al. (2020) extended this paradigm
cross-lingually with XLM-R, pretraining a single model across 100
languages simultaneously (Kinyarwanda among them) rather than requiring a
separate monolingual model per language — the specific capability this
project depends on, since a Kinyarwanda-only pretrained model of
comparable quality does not exist.

Using a multilingual, general-purpose pretrained model for a narrow,
domain-specific task is itself an established technique rather than an
improvised choice. Jørgensen et al. (2021) demonstrate multilingual
domain-adaptive pretraining (MDAPT) — continuing a multilingual model's
pretraining on in-domain text before fine-tuning on the downstream task —
across biomedical and financial domains, showing that this narrows the
gap between general multilingual models and domain-specialized
monolingual ones, particularly for lower-resource languages in their
evaluation set. This project's fine-tuning of XLM-R directly on
commerce-domain messages, rather than first continuing pretraining on
a separately assembled corpus of general commerce text, is a lighter-weight
variant of the same underlying idea; Pakhale's (2023) broader survey of
NER methodology situates this choice within the current landscape of
domain-specific NER approaches, from rule-based systems through
BERT-style architectures to more recent prompt- and retrieval-augmented
methods.

## 3. African-language and Kinyarwanda-specific NLP

MasakhaNER (Adelani et al., 2021, 2022) is the direct precedent this
project extends: the first large, high-quality NER dataset spanning ten
African languages, including Kinyarwanda, and the study that established
XLM-R as the strongest available multilingual baseline for the task. Its
scope, however, is explicitly news-domain text — a formal register with
consistent capitalization, punctuation, and sentence structure that bears
little resemblance to how a Duka shopkeeper actually texts about a sale.
Shiaki's (2025) more recent benchmarking work extends African NER
evaluation explicitly into code-switched and dialectal text, signalling
that the field has recognized this same formality gap; this project's
contribution is to close it specifically for commerce-domain,
transaction-describing text, a register neither MasakhaNER nor this newer
benchmark directly covers.

Kinyarwanda-specific NLP work outside the NER task confirms both that the
underlying data resources for the language are improving and that
domain/register mismatches persist across modalities. Nzeyimana's (2023)
KinSPEAK applies semi-supervised learning to Kinyarwanda automatic speech
recognition, achieving state-of-the-art word error rate on the Mozilla
Common Voice benchmark — evidence that Kinyarwanda language technology
is an active, improving research area, not a neglected one. Most directly
relevant of all is Lester et al.'s (2025) study applying NLP topic
classification to a national-scale corpus of real Kinyarwanda-majority
SMS conversations between COVID-19 patients and healthcare providers in
Rwanda (67% Kinyarwanda, 25% English, the remainder mixed). This is the
closest existing precedent to this project's actual data: real,
informal, short-form Kinyarwanda text messages, processed with NLP at
scale, in a Rwandan public-service context. Its existence demonstrates
both that this kind of work is feasible and valuable, and that — to this
project's knowledge — no equivalent has been built for the commerce
domain specifically.

## 4. Annotation methodology and evaluation statistics

Building a new annotated dataset, rather than reusing an existing one,
raises its own methodological questions that this project's data
collection and evaluation protocol draw on directly. Sabou et al. (2014)
propose best-practice guidelines for corpus annotation, addressing task
decomposition, contributor management, and quality control — directly
relevant to structuring the second-annotator protocol used to compute
inter-annotator agreement in this project (`docs/ANNOTATION_GUIDE.md`).
That agreement is reported as Cohen's Kappa and interpreted against the
Landis and Koch (1977) scale, the standard reference scale for
categorical agreement statistics and the same one used in the MasakhaNER
papers this project's NLP work builds on, making the reported number
directly comparable to published Kinyarwanda NER research.

Evaluating model comparisons statistically, rather than by raw metric
values alone, is a second methodological thread running through both of
this project's research questions. Demšar (2006) argues for
non-parametric statistical tests (Wilcoxon signed-rank, Friedman) when
comparing classifiers across datasets, rather than relying on point
estimates of accuracy or F1 — the same underlying principle motivating
this project's use of the Diebold-Mariano test (Diebold & Mariano, 1995)
for the forecasting benchmark, applied here to the qualitatively
different question of whether one NLP model's extraction quality is
reliably better than another's, not just numerically higher on one split.

## 5. Digital and financial inclusion in informal-sector Africa

The motivation for delivering this system through WhatsApp and USSD,
rather than a standalone app, rests on a substantial body of evidence
that mobile-mediated services are the primary channel through which
financial and commercial inclusion has actually reached underserved
populations in Africa. Suri and Jack's (2016) landmark study of Kenya's
M-PESA found that access to mobile money lifted an estimated 194,000
households out of poverty, with effects concentrated among
female-headed households and driven by improved financial resilience and
occupational mobility — the foundational evidence that mobile-delivered
financial tools produce real economic outcomes, not just convenience.
Grzybowski, Lindlacher, and Mothobi (2023) extend this picture across
nine Sub-Saharan African countries, linking mobile money adoption to
network coverage and demographic factors at a regional scale, while Ky
and Rugemintwari (2025) narrow the focus to exactly this project's target
population — informal firms — finding that Burkinabé informal
enterprises' mobile money adoption patterns shift under economic
uncertainty in ways formal-sector studies do not capture.

The choice of USSD as a parallel channel to WhatsApp is grounded directly
in Rwanda's own household data. The National Institute of Statistics of
Rwanda's EICV7 survey (2023–2024) records mobile phone ownership at 85%
of Rwandan households against smartphone ownership of only 34% — a gap
of over 50 percentage points that would exclude the majority of
households from a WhatsApp-only design, and the direct empirical
justification for this project's three-channel architecture rather than
a smartphone-only one.

## 6. Conversational commerce and chatbots for small business

Deploying a conversational interface for a small or micro-enterprise is
an established but still-maturing research area. Bavaresco et al.'s
(2020) systematic review of conversational agents in business covers a
decade of literature on the goals, domains, and computational methods
behind business-facing chatbots, but notes — as a stated gap — that
studies combining self-learning, personalization, and generative response
methods for a single deployed business solution remain rare. Cordero,
Barba-Guamán, and Guamán's more targeted study of chatbots specifically
in micro, small, and medium enterprises (MSMEs) makes the same
observation from the opposite direction: most chatbot research targets
large enterprises with dedicated technical resources, while
micro-enterprise deployments — this project's actual context — receive
comparatively little attention despite representing the majority of
businesses in many economies.

The closest direct precedent, geographically and in scale of business, is
Azinya and Mashigo's (2025) qualitative study of WhatsApp Business
adoption among South African small and micro-retailers, which identifies
factors like consistent communication frequency and platform
accessibility as drivers of successful adoption among exactly this
population — small African retail businesses using WhatsApp as their
primary digital channel. Their study, however, examines WhatsApp Business
as a human-operated communication tool; it does not involve automated
natural-language understanding of transaction content, which is where
this project's contribution sits.

## 7. Time-series forecasting under data scarcity (dropped from the final report's scope, 2026-07-19)

**This entire section is retained here only as a record of the sources
used while the forecasting question was still part of the project scope.
It is not part of the final capstone report's literature review** — the
report was rescoped to the NLP question alone once that question reached
a complete real result (F1 0.959, Kappa 0.951, 91.5% annotation coverage).
The forecasting code, models, and this literature remain accurate to the
repository; they are simply outside what the thesis argues or defends.

This project's forecasting benchmark evaluates four model classes —
SARIMA, Prophet (Taylor & Letham, 2018), XGBoost (Chen & Guestrin, 2016),
and N-BEATS (Oreshkin et al., 2020) — against a naive baseline under
simulated cold-start conditions, following the walk-forward validation
methodology established by Bergmeir and Benitez (2012) and extended to
demand forecasting specifically by Suradhaniwar et al. (2021), since
standard random-split cross-validation leaks future information into
training when applied to time-ordered data. Falatouri et al. (2022) and
Fatima and Salam (2025) motivate the specific model selection: SARIMA's
competitiveness on shorter, clearly seasonal series, and XGBoost's
documented strength on sparse, intermittent retail demand respectively —
the regime a cold-start Duka shop's sales data actually resembles. More
recent work continues to treat cold-start forecasting as an open problem
rather than a solved one: Zhou et al.'s 2026 application of conditional
diffusion models to new-product cold-start forecasting is one of several
current approaches (alongside transfer learning and meta-learning
methods) being explored as alternatives to the classical and
gradient-boosted methods this project benchmarks directly.

---

## Summary: the gap this project fills

The general methodology exists and is well-established — transformer
fine-tuning for low-resource NER, mobile-delivered inclusion tools for
feature-phone users — but its specific combination — applied to informal,
code-switched Kinyarwanda commerce text, for Rwandan micro-retailers,
delivered through the channels they actually have access to — does not
appear in the existing literature. MasakhaNER answers "can XLM-R handle
Kinyarwanda NER" for news text; this project asks the same question for
commerce text, and answers it: F1 0.959 against a 0.549 rule-based
baseline, on real annotated data with 0.951 inter-annotator agreement.
