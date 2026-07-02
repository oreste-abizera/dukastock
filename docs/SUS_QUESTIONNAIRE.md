# System Usability Scale (SUS) Questionnaire

Addresses proposal Research Question 3 (tertiary) and Specific Objective 3
("...conduct System Usability Scale feedback from a structured
demonstration session"). Administer this to each Duka operator participant
immediately after they finish a hands-on demonstration session with the
WhatsApp and/or USSD prototype — not from memory days later.

## Standard SUS wording (English) — do not alter

The System Usability Scale (Brooke, 1996) is a validated 10-item
instrument. Per standard SUS administration guidance, the wording below
must not be changed and the question order must not be altered, or the
result is no longer comparable to the published benchmark norms (average
SUS score across hundreds of published studies: 68/100).

Each item is rated on a 5-point scale: **1 = Strongly Disagree, 2 =
Disagree, 3 = Neutral, 4 = Agree, 5 = Strongly Agree.**

| # | Statement |
|---|---|
| 1 | I think that I would like to use this system frequently. |
| 2 | I found the system unnecessarily complex. |
| 3 | I thought the system was easy to use. |
| 4 | I think that I would need the support of a technical person to be able to use this system. |
| 5 | I found the various functions in this system were well integrated. |
| 6 | I thought there was too much inconsistency in this system. |
| 7 | I would imagine that most people would learn to use this system very quickly. |
| 8 | I found the system very cumbersome to use. |
| 9 | I felt very confident using the system. |
| 10 | I needed to learn a lot of things before I could get going with this system. |

"This system" refers to DukaStock — be specific with participants about
whether you mean the WhatsApp channel, the USSD channel, or both combined,
and ask the same set of 10 questions once per channel if a participant
tried both, so per-channel comparison is possible (the proposal's research
question asks about "the WhatsApp and USSD prototype channels," plural).

## Kinyarwanda translation

Provided for participants more comfortable responding in Kinyarwanda.
**Important limitation to state explicitly in the thesis:** translating a
validated psychometric instrument is not the same as re-validating it in
the target language. This translation has not been through a formal
back-translation and validation process (the standard rigor for adapting
SUS into a new language). Treat Kinyarwanda-collected scores as
directionally informative alongside the English scores, not as strictly
equivalent to the English-language 68-point published benchmark.

Likert scale: **1 = Sinemeranya na gato, 2 = Simeranya, 3 = Ndabona hagati, 4 = Ndemeranya, 5 = Ndemeranya cyane.**

| # | Statement |
|---|---|
| 1 | Ndatekereza ko nakwifuza gukoresha iyi sisitemu kenshi. |
| 2 | Nasanze iyi sisitemu igoye ku buryo budakenewe. |
| 3 | Natekereje ko iyi sisitemu yoroshye gukoresha. |
| 4 | Ndatekereza ko nakenera ubufasha bw'umuhanga mu ikoranabuhanga kugira ngo nkoreshe iyi sisitemu. |
| 5 | Nasanze ibice bitandukanye by'iyi sisitemu byari bihuye neza. |
| 6 | Natekereje ko hari ibinyuranye byinshi muri iyi sisitemu. |
| 7 | Ndatekereza ko abantu benshi bashobora kwiga gukoresha iyi sisitemu vuba. |
| 8 | Nasanze iyi sisitemu igoye cyane gukoresha. |
| 9 | Numvise nizeye cyane mu gukoresha iyi sisitemu. |
| 10 | Nakeneye kwiga ibintu byinshi mbere yo gutangira gukoresha iyi sisitemu. |

## Administration protocol

1. **Minimum participants**: the proposal scope (Chapter 1.5) specifies a
   minimum of 3 Kigali Duka operators for the demonstration session. SUS's
   own literature notes it remains reliable even with small samples (as
   few as 8-12 per condition for formal benchmarking; smaller samples like
   3-5 are common and acceptable for formative, exploratory usability
   feedback, which is the proposal's framing — just don't claim
   statistical generalizability from n=3).
2. **Timing**: administer immediately after the hands-on demonstration,
   not afterward from memory.
3. **No interruption**: ask all 10 questions together, without other
   questions mixed in between them. You can ask other things (open-ended
   feedback, demographic questions) before or after the SUS block, just
   not interleaved with it.
4. **No explaining the questions**: if a participant asks what a question
   means, give the same brief clarification to every participant — don't
   improvise different explanations per person, which would bias results.
5. **One pass per channel tested**: if a participant tries both WhatsApp
   and USSD, have them fill out the SUS block once per channel, with the
   channel named explicitly ("Mu byerekeye ubu buryo bwa WhatsApp..." /
   "regarding the WhatsApp version...").
6. **Record raw 1-5 responses per item**, not pre-computed scores — let
   `ml_experiments/scripts/score_sus.py` do the scoring consistently.

## Data format for scoring

Save responses as a CSV with one row per participant per channel:

```csv
participant_id,channel,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10
P1,whatsapp,4,2,5,1,4,2,5,1,4,2
P2,whatsapp,5,1,4,2,5,1,4,2,5,2
P3,ussd,3,3,3,3,3,3,3,3,3,3
```

`channel` should be `whatsapp`, `ussd`, or another label if you test
additional variants. Pass this file to
`ml_experiments/scripts/score_sus.py` — see that script's own docstring
for usage and output.

## Reference

Brooke, J. (1996). SUS: A "quick and dirty" usability scale. In P. W.
Jordan, B. Thomas, B. A. Weerdmeester, & I. L. McClelland (Eds.), Usability
Evaluation in Industry (pp. 189-194). Taylor & Francis. The 10-item
instrument and 1986-original Likert wording reproduced above match the
standard form documented across SUS implementation guides (e.g. Sauro &
Lewis methodology; UXtweak, CleverX, Trymata practitioner guides, 2024-2026).
