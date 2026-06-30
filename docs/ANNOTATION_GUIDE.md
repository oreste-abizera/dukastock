# Annotation Guide: Commerce-Domain Kinyarwanda NER Test Set

This guide covers collecting and annotating the 200-message test set used
in `ml_experiments/notebooks/03_xlmr_commerce_ner.ipynb` and
`ml_experiments/scripts/train_xlmr_ner.py` (proposal Chapter 1.3.1 /
Chapter 3.3, Research Question 2). It replaces the synthetic placeholder
data those files generate when no real annotation file is present.

If you only do one thing from this document, do this: **collect real
messages before you annotate them.** A dataset of invented sentences,
however carefully you write them, won't carry the spelling
inconsistencies, code-switching patterns, and shorthand that real Duka
shopkeepers actually use — which is the entire point of the test set.

---

## 1. What you're collecting

200 short, informal messages describing a sales transaction, in the mix of
Kinyarwanda, English, and French-derived loanwords a Duka shopkeeper would
actually type on WhatsApp. Each message gets three entity types marked:

| Entity | What it marks | Examples |
|---|---|---|
| `PRODUCT` | The FMCG item sold | isukari, amavuta, ifu, umuceri, isabune, sugar, oil |
| `QUANTITY` | The number sold | bitatu, rimwe, 3, 5 |
| `UNIT` | The unit of measure | ibiro, litre, kg, ipande |

A message can have more than one of each (a shopkeeper reporting two sales
at once), or be missing one (someone forgets to mention the unit). Annotate
what's actually in the message — don't infer or add what you think they
meant.

### Where to get real messages

In rough order of how representative they'll be:

1. **Actual WhatsApp messages from pilot Duka shopkeepers**, if you have
   any from earlier outreach (e.g. through the Growthwave/ALU network) —
   by far the best source, since this is exactly the deployment population.
2. **Elicited messages**: ask 10-15 shopkeepers to text you "what they sold
   today" in their own words, framed as a quick favor rather than a formal
   survey. Different people will phrase it differently, which is what you
   want.
3. **Self-generated, but written by multiple different people**, each
   asked independently to imagine texting a friend about today's sales —
   weaker than 1 or 2, but still better than one person inventing all 200
   in a single sitting (which tends to converge on one phrasing style and
   under-represents real variation).

Aim for variety on purpose: mix Kinyarwanda-only, English-only, and
code-switched messages; mix digit quantities ("3") with number words
("bitatu"); include a few messages with typos, missing units, or unusual
word order, since those are exactly the cases the RapidFuzz baseline will
fail on and where XLM-R needs to prove it does better.

### How many messages, and the train/eval split

200 total, matching the proposal's stated test set size.
`train_xlmr_ner.py` and Notebook 3 both split this 80/20 automatically
(160 train / 40 eval) — you don't need to pre-split the file yourself,
just hand over all 200 in one file per annotator.

---

## 2. Annotation tool: Doccano

[Doccano](https://github.com/doccano/doccano) is a free, open-source,
actively maintained web-based annotation tool built specifically for
sequence labeling tasks like this one. It runs locally via Docker, so no
data leaves your machine.

### Setup

```bash
docker pull doccano/doccano
docker container create --name doccano \
  -e "ADMIN_USERNAME=admin" \
  -e "ADMIN_EMAIL=admin@example.com" \
  -e "ADMIN_PASSWORD=<choose-a-password>" \
  -p 8000:8000 doccano/doccano
docker container start doccano
```

Then open `http://localhost:8000` and log in.

### Project setup

1. Create a project, type **"Sequence Labeling"**.
2. Under **Labels**, create three labels: `PRODUCT`, `QUANTITY`, `UNIT`.
   Give each a distinct color and a keyboard shortcut (Doccano supports
   this) — it speeds up annotation a lot once you're doing 200 messages.
3. Prepare your 200 messages as a plain text file (one message per line)
   or a simple JSON list, then **Import Dataset** into the project.
4. Annotate by selecting a text span with your mouse and clicking the
   matching label (or using its shortcut key).
5. When done, go to **Export Dataset** and choose the **JSONL** export
   option (Doccano calls this "JSONL(Text-Labels)" in current versions, or
   plain "JSONL" in older ones — either is fine, the converter below
   handles both).

### Converting Doccano's export to DukaStock's format

Doccano's exported field names have changed across versions and export
button choices, and none of them match the schema `train_xlmr_ner.py`
expects out of the box. Use the converter script included in this repo
rather than hand-editing the export:

```bash
python ml_experiments/scripts/convert_doccano_export.py \
  --input path/to/doccano_export.jsonl \
  --output ml_experiments/data/annotations.jsonl
```

This produces the exact schema `train_xlmr_ner.py` and Notebook 3 expect:

```json
{"text": "Nabagurishije isukari ibiro bitatu",
 "entities": [{"start": 14, "end": 21, "label": "PRODUCT"},
              {"start": 22, "end": 27, "label": "UNIT"},
              {"start": 28, "end": 35, "label": "QUANTITY"}]}
```

The converter handles three Doccano export shapes (older `labels` format,
newer `data`/`label` format, and the `entities`/`start_offset` format) and
will print a warning and skip — rather than crash on — any malformed line,
so a handful of export glitches won't block the whole conversion.

---

## 3. Second annotator protocol (for Cohen's Kappa)

The proposal's data quality protocol calls for inter-annotator agreement,
reported as Cohen's Kappa — the same metric the MasakhaNER papers
(Adelani et al., 2021, 2022) use, so your number is directly comparable to
published Kinyarwanda NER work.

### Steps

1. **Pick a subset, not necessarily all 200.** A second full annotation
   pass of all 200 messages is ideal if you have the time; if not, even
   50 messages is enough to compute a meaningful Kappa. Whatever subset
   you choose, both annotators must annotate the *exact same* messages.
2. **Annotator B works independently**, with no access to Annotator A's
   labels, but with the *same* label definitions (Section 1 above). Don't
   let them discuss specific messages until after both are done — that
   defeats the point of measuring independent agreement.
3. **Give Annotator B the raw, unlabeled messages** — export the plain
   text list from your Doccano project (or just hand over the original
   message file) and have them set up their own Doccano project, or a
   second Doccano account/workspace, to annotate independently.
4. **Export and convert Annotator B's file the same way** as Annotator A's:

   ```bash
   python ml_experiments/scripts/convert_doccano_export.py \
     --input annotator_b_export.jsonl \
     --output ml_experiments/data/annotations_annotator_b.jsonl
   ```

5. **Compute Kappa:**

   ```bash
   python ml_experiments/scripts/compute_kappa.py \
     --annotator-a ml_experiments/data/annotations.jsonl \
     --annotator-b ml_experiments/data/annotations_annotator_b.jsonl
   ```

   This prints the Kappa score, an interpretation against the standard
   Landis & Koch (1977) scale, and warns you about any messages one
   annotator skipped or where the two found a different number of
   entities (these mismatches are usually the most informative — go look
   at them).

### Resolving disagreements

If Kappa comes back below ~0.4 ("fair" or worse), don't proceed straight
to fine-tuning — the label definitions are probably ambiguous in a way
that will also confuse the model. Common fixes:

- **Unit ambiguity**: is "amavuta" (oil, no unit given) tagged as PRODUCT
  only, or does the annotator guess a unit? Pick one rule and document it.
- **Compound products**: if a message mentions two products, are they two
  separate PRODUCT spans or does only the first count? (Two separate spans
  is the convention used throughout this codebase — see
  `app/nlp/ner_pipeline.py`.)
- **Number words vs. digits**: make sure both annotators tag QUANTITY the
  same way whether it's written as a word ("bitatu") or a digit ("3") —
  both should be tagged identically as QUANTITY.

Re-annotate a small calibration batch together after agreeing on the fix,
then proceed with the second annotator's full independent pass.

---

## 4. After annotation: what happens next

Once `ml_experiments/data/annotations.jsonl` exists with real data:

- `ml_experiments/notebooks/03_xlmr_commerce_ner.ipynb` will detect the
  file automatically (`USING_SYNTHETIC_DATA` will print `False`) and use
  it instead of generating placeholder messages.
- `ml_experiments/scripts/train_xlmr_ner.py` reads the same file by
  default (`--annotations` flag, defaults to this path).
- Re-run Notebook 3 end-to-end; the precision/recall/F1 numbers and the
  Cohen's Kappa it reports are now real results, safe to write into the
  thesis — at that point you can remove the synthetic-data warnings from
  your written discussion.

See `docs/SOURCES.md` for how the synthetic-data fallback is flagged
elsewhere in this codebase, and `docs/RESEARCH_DESIGN.md` for how
Research Question 2 frames this evaluation.
