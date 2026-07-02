# Data Sources Research: Kinyarwanda-English Commerce NER
*For DukaStock Notebook 3 — 200-message annotated dataset*

---

## The honest finding

**There is no existing public dataset of informal Kinyarwanda-English code-switched WhatsApp commerce messages.** This is not a gap in my search — it is the novelty of your thesis. Every Kinyarwanda NLP dataset published to date covers either news text (MasakhaNER), speech (Common Voice), or formal government documents (Rwandan Gazette). The commerce-domain, code-switched register you are studying has not been annotated before. That's your contribution.

What I found is a set of raw text sources you can mine, plus a validated strategy for generating the dataset you need right now.

---

## Source 1: MasakhaNER (Kinyarwanda) — Hugging Face

**URL:** https://huggingface.co/datasets/masakhane/masakhaner2  
**Access:** Free, instant  
**Load:** `load_dataset("masakhane/masakhaner2", "kin")`

This is the gold standard for Kinyarwanda NER, cited in your thesis proposal. The entity types are **PER, ORG, LOC, DATE** — not PRODUCT/QUANTITY/UNIT — so it cannot be used directly as training data. However, it is useful for two things:

- **Vocabulary reference:** The raw sentences contain realistic Kinyarwanda sentence structure, helping calibrate what "natural" Kinyarwanda text looks like in an annotated context.
- **Transfer learning baseline:** The `mbeukman/xlm-roberta-base-finetuned-kinyarwanda-finetuned-ner-kinyarwanda` model on HuggingFace was pre-trained on this data and can be used as a starting checkpoint before domain adaptation.

---

## Source 2: mbazaNLP Kinyarwanda Monolingual Corpus — Hugging Face

**URL:** https://huggingface.co/datasets/mbazaNLP/kinyarwanda_monolingual_v01.1  
**Access:** Free, CC BY 4.0  
**Size:** 78,000 documents, 25 million words

This is the largest open Kinyarwanda text dataset. Sources include Kigali Today, Igihe, Wikipedia, government reports, and cultural narratives. Use the **v01.1** version (v01.0 has duplicates).

**How to mine it for your use case:**

```python
from datasets import load_dataset
import re

ds = load_dataset("mbazaNLP/kinyarwanda_monolingual_v01.1", split="train")

COMMERCE_WORDS = {
    "isukari", "amavuta", "ifu", "umuceri", "isabune",
    "ibigori", "ibishyimbo", "amata", "ikawa", "icyayi",
    "ibiro", "litre", "litro", "ipande", "agasanduku",
    "nabagurishije", "naragurishije", "byagurishijwe",
    "acuruza", "kugura", "gusunika",
}

def is_commerce_sentence(text):
    words = set(text.lower().split())
    return len(words & COMMERCE_WORDS) >= 2

candidates = [row["text"] for row in ds if is_commerce_sentence(row["text"])]
```

I can already see one sentence in the corpus preview: *"Ibicuruzwa ahanini bivugwa kubamo ubujura ni umuceri, kawunga, isukari n'amavuta yo guteka..."* — this is real commerce language. Filtering this corpus will yield 50–100 genuine Kinyarwanda sentences mentioning products and transactions, which you can annotate manually in an hour.

---

## Source 3: Leipzig Corpora Collection (Kinyarwanda Community 2017) — Free Download

**URL:** https://wortschatz.uni-leipzig.de/en/download/Kinyarwanda  
**Access:** Free, no account needed  
**Size:** 54,359 sentences (community web text)

Plain `.txt` download, no API required. This corpus was collected from community web sources (forums, social media-adjacent text), so it is closer to informal register than news corpora. Apply the same vocabulary filter as Source 2 to extract commerce-adjacent sentences.

---

## Source 4: WFP Rwanda Food Prices (HDX) — CSV

**URL:** https://data.humdata.org/dataset/wfp-food-prices-for-rwanda  
**Access:** Free, CC BY-IGO  
**Format:** CSV updated weekly

This is NOT text data. It is a structured table of commodity prices tracked by the World Food Programme across Rwanda's markets (Kimironko, Nyamirambo, Musanze, etc.). Its value for your project:

- **Ground truth vocabulary:** Real product names (Sorghum, Cassava flour, Groundnut oil, Dried beans...) and real market names to use in synthetic generation
- **Price realism:** Real price ranges for RWF amounts when you add price context to messages
- **Thesis credibility:** You can cite WFP data as grounding your product vocabulary choices

---

## Source 5: DigitalUmuganda / Rwandan Official Gazette — Hugging Face

**URL:** https://huggingface.co/datasets/DigitalUmuganda/NMT_Rwandan-Gazette_parallel_data_en_kin  
**Access:** Free  

Parallel Kinyarwanda-English sentences extracted from the Official Gazette of Rwanda. Formal register, mostly legal/procurement text. Some sentences mention quantities and products in procurement contexts (useful for the formal end of register variation).

---

## What does NOT exist (and why that matters for your thesis)

- No WhatsApp commerce message dataset in Kinyarwanda
- No code-switched KW-EN trade message dataset for any African language pair
- No informal market-domain NER corpus for Rwanda

This gap is the stated motivation of your research question. Citing these source searches in your Methods chapter (Section 3.3 Data Collection) strengthens the case for why you had to create new annotations rather than reuse existing ones.

---

## The pragmatic path: a synthetic placeholder, not a substitute

**Read this before doing anything with the generated file.** Given the time
constraint, the vocabulary and structure from the existing notebook were
extended into a template generator that produces 200 messages with
intentional structural variation. But "verified with 0 span errors" below
means the character offsets match the text the generator itself inserted —
it does **not** mean a human confirmed the labels are correct, because no
human looked at these messages. This is programmatic label computation,
not annotation. It cannot stand in for the real 200-message annotated test
set your proposal promises, and it cannot produce a meaningful Cohen's
Kappa (there is no independent second annotator — see below).

**File:** `annotations_SYNTHETIC_placeholder.jsonl` (in the same folder as
this document). The filename is deliberately unambiguous.
**Verification:** All 200 records pass a self-consistency check (character
offsets match the inserted text) — this is a code-correctness check on the
generator, not a validity check on the data.

**Variation statistics:**
| Dimension | Coverage |
|---|---|
| Kinyarwanda-flavored messages | ~120 |
| English-flavored messages | ~74 |
| True code-mixed (KW verb + EN product or vice versa) | ~60 |
| Digit quantities (e.g., "3") | ~100 |
| Kinyarwanda word quantities (e.g., "bitatu") | ~70 |
| English word quantities (e.g., "three") | ~30 |
| Messages missing UNIT (realistic) | 9 |
| Messages missing QUANTITY (realistic) | 1 |
| Messages with two products (e.g., "isukari na amavuta") | 5 |
| Messages with market context (Kimironko, Remera…) | ~12 |
| Messages with price (RWF/Frw) | ~24 |
| Kinyarwanda products | 14 distinct |
| English products | 15 distinct |
| Kinyarwanda number words | 15 distinct |
| Unit types | 15 distinct |
| Sentence templates | 30+ patterns |

---

## How NOT to use the generated file

**Do not copy or rename this file to `ml_experiments/data/annotations.jsonl`.**
That path is what Notebook 3 / `train_xlmr_ner.py` treat as the real test
set — dropping the synthetic file there makes `USING_SYNTHETIC_DATA` print
`False`, which would misrepresent every downstream number (P/R/F1, Kappa)
as coming from real annotated data when it does not. The generator output
is useful only as: (a) a way to pipeline-test the notebook/training script
end-to-end before real data exists, or (b) a source of candidate sentence
*templates* that a human annotator collecting real messages could draw
inspiration from — never as a drop-in replacement for the dataset itself.

For the real test set, follow `docs/ANNOTATION_GUIDE.md`: collect actual
messages (ideally from real Duka shopkeepers, per the guide's sourcing
section), have them independently labeled by two human annotators, and
place the result at `ml_experiments/data/annotations.jsonl`.

**On citing synthetic-data literature:** ArXiv:2505.16814 and similar
2025 work on LLM-generated NER training data are about *augmenting
training sets*, not about substituting for an *evaluation* set used to
answer a research question with reported significance/agreement
statistics. Citing that literature to justify treating this file as the
thesis's test set would be a methodological misapplication — happy to
discuss if there's a training-augmentation use case where it's actually
appropriate, but that's a different claim than what RQ2 asks.

**Cohen's Kappa:** Computing Kappa from two people independently labeling
*these* synthetic sentences would measure agreement on invented text, not
on real Duka shopkeeper language — it does not produce the inter-annotator
agreement figure the proposal's RQ2 asks for. A defensible Kappa requires
two independent annotators labeling the *real* collected messages.

---

## Citation anchors for your Methods chapter

- Adelani et al. (2021, 2022) — MasakhaNER v1 and v2, Kinyarwanda NER baseline
- Mbaza NLP Community (2024) — Kinyarwanda Monolingual Dataset v01.1, HuggingFace
- WFP (2024) — Rwanda Food Prices, HDX, CC BY-IGO
- Leipzig Corpora Collection — kin_community_2017, wortschatz.uni-leipzig.de
- ArXiv:2505.16814 — "Does Synthetic Data Help NER for Low-Resource Languages?" (2025)
- ArXiv:2310.01119 — "Synthetic Data Generation in Low-Resource Settings" (2023)
