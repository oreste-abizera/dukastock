"""
SYNTHETIC DATA — NOT A REAL ANNOTATED TEST SET. DO NOT REPORT IN THE THESIS.

Generates 200 template-based Kinyarwanda-English commerce NER messages with
programmatically-computed (not human-verified) character-offset annotations
for PRODUCT, QUANTITY, UNIT. The labels are correct by construction
(str.find() + a self-assertion that the label matches what was just
inserted) — this is span-computation, not annotation. There is no
independent human judgment anywhere in this file's output, so it cannot
produce meaningful precision/recall/F1 against a "ground truth" it wrote
itself, and it cannot be used to compute a real Cohen's Kappa (there is no
second annotator).

This script exists only to pipeline-test Notebook 3 / train_xlmr_ner.py
before real data exists (see USING_SYNTHETIC_DATA in Notebook 3, which
prints this fact on every figure it produces from this data). For the real
200-message test set, follow docs/ANNOTATION_GUIDE.md — collect actual
messages from Duka shopkeepers (or elicited from real people), have them
independently labeled by two human annotators, and place the result at
ml_experiments/data/annotations.jsonl. Do NOT copy this script's output
there — that would silently make USING_SYNTHETIC_DATA report False.

Intentional variation across:
  - language mix (KW-only, EN-only, code-mixed)
  - number format (KW word, EN digit, EN word)
  - sentence structure (5+ template families)
  - product vocabulary (14 KW + 15 EN products)
  - missing entities (some messages lack UNIT or QUANTITY)
  - two products in one message
  - occasional realistic typos
"""

import json, random
from pathlib import Path

random.seed(42)

# ── Vocabulary ──────────────────────────────────────────────────────────────

PRODUCTS_KW = [
    "isukari", "amavuta", "ifu", "umuceri", "isabune",
    "inzoga", "ibigori", "ibishyimbo", "amata", "ikawa",
    "icyayi", "inyama", "ibitoki", "imiteja",
]
PRODUCTS_EN = [
    "sugar", "oil", "flour", "rice", "soap",
    "beer", "maize", "beans", "milk", "coffee",
    "tea", "meat", "bananas", "salt", "tomatoes",
]
NUMBERS_KW = [
    "rimwe", "kabiri", "bitatu", "bine", "bitanu",
    "esheshatu", "birindwi", "umunani", "icyenda", "icumi",
    "imwe", "ebyiri", "eshatu", "enye", "eshanu",
]
NUMBERS_EN = ["one", "two", "three", "four", "five", "six", "seven", "eight", "ten", "twelve"]
NUMBERS_DIG = ["1", "2", "3", "4", "5", "6", "7", "8", "10", "12", "15", "20", "25", "50"]
UNITS_KW = ["ibiro", "litre", "litro", "ipande", "agasanduku", "indobo", "ifurishi"]
UNITS_EN = ["kg", "kilo", "liters", "bar", "box", "bottle", "bag", "sack"]

# Map KW product → EN equivalent for label consistency
KW_TO_CANONICAL = dict(zip(PRODUCTS_KW, PRODUCTS_EN))

# Typo variants (realistic misspellings seen in WhatsApp)
TYPOS = {
    "isukari":  "sukari",
    "isabune":  "sabune",
    "amavuta":  "mavuta",
    "umuceri":  "umucely",
    "ibigori":  "ibigoly",
    "ibishyimbo": "ibishyimbu",
    "ibiro":    "ibilo",
    "litre":    "liter",
    "agasanduku": "gasanduku",
}

def maybe_typo(word, rate=0.08):
    if random.random() < rate and word in TYPOS:
        return TYPOS[word]
    return word

# ── Span helper ──────────────────────────────────────────────────────────────

def make_entity(text, word, label, search_from=0):
    idx = text.find(word, search_from)
    if idx == -1:
        return None
    return {"start": idx, "end": idx + len(word), "label": label, "_w": word}

def verify_and_clean(text, entities):
    out = []
    for e in entities:
        if e is None:
            continue
        extracted = text[e["start"]:e["end"]]
        assert extracted == e["_w"], f"SPAN ERROR: got '{extracted}', expected '{e['_w']}' in:\n  {text}"
        out.append({"start": e["start"], "end": e["end"], "label": e["label"]})
    return {"text": text, "entities": out}

# ── Template families ────────────────────────────────────────────────────────

def pick_product(lang="kw"):
    if lang == "kw":
        p = random.choice(PRODUCTS_KW)
        return maybe_typo(p)
    else:
        return random.choice(PRODUCTS_EN)

def pick_qty(fmt="kw"):
    if fmt == "kw":   return random.choice(NUMBERS_KW)
    if fmt == "en":   return random.choice(NUMBERS_EN)
    return random.choice(NUMBERS_DIG)

def pick_unit(lang="kw"):
    u = random.choice(UNITS_KW if lang == "kw" else UNITS_EN)
    return maybe_typo(u)

# ── Generator functions (each returns one record) ───────────────────────────

def tmpl_kw_sold():
    """Nabagurishije [PRODUCT] [UNIT] [QUANTITY]"""
    p = pick_product("kw")
    u = pick_unit("kw")
    q = pick_qty("kw")
    text = f"Nabagurishije {p} {u} {q}"
    ents = [
        make_entity(text, p, "PRODUCT"),
        make_entity(text, u, "UNIT"),
        make_entity(text, q, "QUANTITY"),
    ]
    return verify_and_clean(text, ents)

def tmpl_kw_sold_qty_first():
    """Naragurishije [PRODUCT] [QUANTITY] [UNIT]"""
    p = pick_product("kw")
    q = pick_qty("kw")
    u = pick_unit("kw")
    text = f"Naragurishije {p} {q} {u}"
    ents = [
        make_entity(text, p, "PRODUCT"),
        make_entity(text, q, "QUANTITY"),
        make_entity(text, u, "UNIT"),
    ]
    return verify_and_clean(text, ents)

def tmpl_kw_passive():
    """[PRODUCT] [QUANTITY] [UNIT] byagurishijwe"""
    p = pick_product("kw")
    q = pick_qty(random.choice(["kw","dig"]))
    u = pick_unit("kw")
    text = f"{p} {q} {u} byagurishijwe"
    ents = [
        make_entity(text, p, "PRODUCT"),
        make_entity(text, q, "QUANTITY"),
        make_entity(text, u, "UNIT"),
    ]
    return verify_and_clean(text, ents)

def tmpl_kw_no_unit():
    """Nabagurishije [PRODUCT] [QUANTITY]  (missing UNIT)"""
    p = pick_product("kw")
    q = pick_qty(random.choice(["kw","dig"]))
    text = f"Nabagurishije {p} {q}"
    ents = [
        make_entity(text, p, "PRODUCT"),
        make_entity(text, q, "QUANTITY"),
    ]
    return verify_and_clean(text, ents)

def tmpl_kw_no_qty():
    """[PRODUCT] [UNIT] byagurishijwe  (missing QUANTITY)"""
    p = pick_product("kw")
    u = pick_unit("kw")
    text = f"{p} {u} byagurishijwe"
    ents = [
        make_entity(text, p, "PRODUCT"),
        make_entity(text, u, "UNIT"),
    ]
    return verify_and_clean(text, ents)

def tmpl_en_sold():
    """Sold [QUANTITY] [UNIT] of [PRODUCT] today"""
    p = pick_product("en")
    q = pick_qty(random.choice(["dig","en"]))
    u = pick_unit("en")
    text = f"Sold {q} {u} of {p} today"
    ents = [
        make_entity(text, p, "PRODUCT"),
        make_entity(text, q, "QUANTITY"),
        make_entity(text, u, "UNIT"),
    ]
    return verify_and_clean(text, ents)

def tmpl_en_sold_v2():
    """Today I sold [QUANTITY] [UNIT] of [PRODUCT]"""
    p = pick_product("en")
    q = pick_qty(random.choice(["dig","en"]))
    u = pick_unit("en")
    text = f"Today I sold {q} {u} of {p}"
    ents = [
        make_entity(text, p, "PRODUCT"),
        make_entity(text, q, "QUANTITY"),
        make_entity(text, u, "UNIT"),
    ]
    return verify_and_clean(text, ents)

def tmpl_en_no_unit():
    """Sold [QUANTITY] [PRODUCT] today  (missing UNIT)"""
    p = pick_product("en")
    q = pick_qty("dig")
    text = f"Sold {q} {p} today"
    ents = [
        make_entity(text, p, "PRODUCT"),
        make_entity(text, q, "QUANTITY"),
    ]
    return verify_and_clean(text, ents)

def tmpl_mixed_kw_verb_en_prod():
    """Nabagurishije [PRODUCT_EN] [UNIT_KW] [QUANTITY_KW]"""
    p = pick_product("en")
    u = pick_unit("kw")
    q = pick_qty("kw")
    text = f"Nabagurishije {p} {u} {q}"
    ents = [
        make_entity(text, p, "PRODUCT"),
        make_entity(text, u, "UNIT"),
        make_entity(text, q, "QUANTITY"),
    ]
    return verify_and_clean(text, ents)

def tmpl_mixed_en_verb_kw_prod():
    """I sold [PRODUCT_KW] [QUANTITY_KW] [UNIT_EN]"""
    p = pick_product("kw")
    q = pick_qty("kw")
    u = pick_unit("en")
    text = f"I sold {p} {q} {u}"
    ents = [
        make_entity(text, p, "PRODUCT"),
        make_entity(text, q, "QUANTITY"),
        make_entity(text, u, "UNIT"),
    ]
    return verify_and_clean(text, ents)

def tmpl_mixed_digit_kw():
    """Nabagurishije [PRODUCT_KW] [UNIT_KW] [DIGIT]"""
    p = pick_product("kw")
    u = pick_unit("kw")
    q = pick_qty("dig")
    text = f"Nabagurishije {p} {u} {q}"
    ents = [
        make_entity(text, p, "PRODUCT"),
        make_entity(text, u, "UNIT"),
        make_entity(text, q, "QUANTITY"),
    ]
    return verify_and_clean(text, ents)

def tmpl_mixed_digit_en_unit():
    """Naragurishije [PRODUCT_KW] [DIGIT] [UNIT_EN]"""
    p = pick_product("kw")
    q = pick_qty("dig")
    u = pick_unit("en")
    text = f"Naragurishije {p} {q} {u}"
    ents = [
        make_entity(text, p, "PRODUCT"),
        make_entity(text, q, "QUANTITY"),
        make_entity(text, u, "UNIT"),
    ]
    return verify_and_clean(text, ents)

def tmpl_mixed_two_products():
    """Nabagurishije [PRODUCT1_KW] na [PRODUCT2_KW] [QUANTITY] [UNIT]"""
    p1 = pick_product("kw")
    p2 = pick_product("kw")
    while p2 == p1:
        p2 = pick_product("kw")
    q = pick_qty(random.choice(["kw","dig"]))
    u = pick_unit("kw")
    text = f"Nabagurishije {p1} na {p2} {q} {u}"
    ents = [
        make_entity(text, p1, "PRODUCT"),
        make_entity(text, p2, "PRODUCT", text.find(p2, text.find(p1) + len(p1))),
        make_entity(text, q, "QUANTITY"),
        make_entity(text, u, "UNIT"),
    ]
    return verify_and_clean(text, ents)

def tmpl_mixed_two_products_en():
    """Sold [PRODUCT1_EN] and [PRODUCT2_EN] [QUANTITY] [UNIT]"""
    p1 = pick_product("en")
    p2 = pick_product("en")
    while p2 == p1:
        p2 = pick_product("en")
    q = pick_qty("dig")
    u = pick_unit("en")
    text = f"Sold {p1} and {p2} {q} {u} today"
    ents = [
        make_entity(text, p1, "PRODUCT"),
        make_entity(text, p2, "PRODUCT", text.find(p2, text.find(p1) + len(p1))),
        make_entity(text, q, "QUANTITY"),
        make_entity(text, u, "UNIT"),
    ]
    return verify_and_clean(text, ents)

def tmpl_kw_context():
    """Muri isoko ya Kimironko nabagurishije [PRODUCT] [UNIT] [QUANTITY]"""
    markets = ["Kimironko", "Remera", "Nyamirambo", "Gisozi", "Kicukiro", "Nyabugogo"]
    m = random.choice(markets)
    p = pick_product("kw")
    u = pick_unit("kw")
    q = pick_qty(random.choice(["kw","dig"]))
    text = f"Muri isoko ya {m} nabagurishije {p} {u} {q}"
    ents = [
        make_entity(text, p, "PRODUCT"),
        make_entity(text, u, "UNIT"),
        make_entity(text, q, "QUANTITY"),
    ]
    return verify_and_clean(text, ents)

def tmpl_en_context():
    """At [MARKET] market I sold [QUANTITY] [UNIT] of [PRODUCT]"""
    markets = ["Kimironko", "Remera", "Nyamirambo", "Gisozi", "Kicukiro"]
    m = random.choice(markets)
    p = pick_product("en")
    q = pick_qty("dig")
    u = pick_unit("en")
    text = f"At {m} market I sold {q} {u} of {p}"
    ents = [
        make_entity(text, p, "PRODUCT"),
        make_entity(text, q, "QUANTITY"),
        make_entity(text, u, "UNIT"),
    ]
    return verify_and_clean(text, ents)

def tmpl_kw_price():
    """Nabagurishije [PRODUCT] [UNIT] [QUANTITY] ku [PRICE] Frw"""
    p = pick_product("kw")
    u = pick_unit("kw")
    q = pick_qty(random.choice(["kw","dig"]))
    prices = ["500", "1000", "1500", "2000", "3000", "5000"]
    pr = random.choice(prices)
    text = f"Nabagurishije {p} {u} {q} ku {pr} Frw"
    ents = [
        make_entity(text, p, "PRODUCT"),
        make_entity(text, u, "UNIT"),
        make_entity(text, q, "QUANTITY"),
    ]
    return verify_and_clean(text, ents)

def tmpl_en_price():
    """Sold [QUANTITY] [UNIT] of [PRODUCT] for [PRICE] RWF"""
    p = pick_product("en")
    q = pick_qty("dig")
    u = pick_unit("en")
    prices = ["500", "1000", "1500", "2000", "3000"]
    pr = random.choice(prices)
    text = f"Sold {q} {u} of {p} for {pr} RWF"
    ents = [
        make_entity(text, p, "PRODUCT"),
        make_entity(text, q, "QUANTITY"),
        make_entity(text, u, "UNIT"),
    ]
    return verify_and_clean(text, ents)

def tmpl_kw_question():
    """Ufite [PRODUCT] [UNIT] [QUANTITY]?"""
    p = pick_product("kw")
    u = pick_unit("kw")
    q = pick_qty(random.choice(["kw","dig"]))
    text = f"Ufite {p} {u} {q}?"
    ents = [
        make_entity(text, p, "PRODUCT"),
        make_entity(text, u, "UNIT"),
        make_entity(text, q, "QUANTITY"),
    ]
    return verify_and_clean(text, ents)

def tmpl_en_question():
    """Do you have [QUANTITY] [UNIT] of [PRODUCT]?"""
    p = pick_product("en")
    q = pick_qty("dig")
    u = pick_unit("en")
    text = f"Do you have {q} {u} of {p}?"
    ents = [
        make_entity(text, p, "PRODUCT"),
        make_entity(text, q, "QUANTITY"),
        make_entity(text, u, "UNIT"),
    ]
    return verify_and_clean(text, ents)

def tmpl_kw_remaining():
    """Hasigaye [PRODUCT] [UNIT] [QUANTITY] gusa"""
    p = pick_product("kw")
    u = pick_unit("kw")
    q = pick_qty(random.choice(["kw","dig"]))
    text = f"Hasigaye {p} {u} {q} gusa"
    ents = [
        make_entity(text, p, "PRODUCT"),
        make_entity(text, u, "UNIT"),
        make_entity(text, q, "QUANTITY"),
    ]
    return verify_and_clean(text, ents)

def tmpl_mixed_remaining():
    """Still have [QUANTITY] [UNIT] of [PRODUCT_KW] remaining"""
    p = pick_product("kw")
    q = pick_qty("dig")
    u = pick_unit("en")
    text = f"Still have {q} {u} of {p} remaining"
    ents = [
        make_entity(text, p, "PRODUCT"),
        make_entity(text, q, "QUANTITY"),
        make_entity(text, u, "UNIT"),
    ]
    return verify_and_clean(text, ents)

def tmpl_kw_received():
    """Nakiriye [PRODUCT] [UNIT] [QUANTITY]"""
    p = pick_product("kw")
    u = pick_unit("kw")
    q = pick_qty(random.choice(["kw","dig"]))
    text = f"Nakiriye {p} {u} {q}"
    ents = [
        make_entity(text, p, "PRODUCT"),
        make_entity(text, u, "UNIT"),
        make_entity(text, q, "QUANTITY"),
    ]
    return verify_and_clean(text, ents)

def tmpl_en_received():
    """Received [QUANTITY] [UNIT] of [PRODUCT]"""
    p = pick_product("en")
    q = pick_qty("dig")
    u = pick_unit("en")
    text = f"Received {q} {u} of {p}"
    ents = [
        make_entity(text, p, "PRODUCT"),
        make_entity(text, q, "QUANTITY"),
        make_entity(text, u, "UNIT"),
    ]
    return verify_and_clean(text, ents)

def tmpl_kw_ordered():
    """Nasabye [PRODUCT] [UNIT] [QUANTITY]"""
    p = pick_product("kw")
    u = pick_unit(random.choice(["kw","en"]))
    q = pick_qty(random.choice(["kw","dig"]))
    text = f"Nasabye {p} {u} {q}"
    ents = [
        make_entity(text, p, "PRODUCT"),
        make_entity(text, u, "UNIT"),
        make_entity(text, q, "QUANTITY"),
    ]
    return verify_and_clean(text, ents)

def tmpl_mixed_ordered():
    """I need [QUANTITY] [UNIT] of [PRODUCT_KW]"""
    p = pick_product("kw")
    q = pick_qty(random.choice(["dig","en"]))
    u = pick_unit("en")
    text = f"I need {q} {u} of {p}"
    ents = [
        make_entity(text, p, "PRODUCT"),
        make_entity(text, q, "QUANTITY"),
        make_entity(text, u, "UNIT"),
    ]
    return verify_and_clean(text, ents)

def tmpl_short_kw():
    """[PRODUCT] [QUANTITY] [UNIT] (ultra-short, no verb)"""
    p = pick_product("kw")
    q = pick_qty(random.choice(["kw","dig"]))
    u = pick_unit(random.choice(["kw","en"]))
    text = f"{p} {q} {u}"
    ents = [
        make_entity(text, p, "PRODUCT"),
        make_entity(text, q, "QUANTITY"),
        make_entity(text, u, "UNIT"),
    ]
    return verify_and_clean(text, ents)

def tmpl_short_en():
    """[PRODUCT] [QUANTITY] [UNIT] sold"""
    p = pick_product("en")
    q = pick_qty("dig")
    u = pick_unit("en")
    text = f"{p} {q} {u} sold"
    ents = [
        make_entity(text, p, "PRODUCT"),
        make_entity(text, q, "QUANTITY"),
        make_entity(text, u, "UNIT"),
    ]
    return verify_and_clean(text, ents)

def tmpl_kw_long():
    """Muri uyu munsi nabagurishije [PRODUCT] [UNIT] [QUANTITY] kuri abakiriya benshi"""
    p = pick_product("kw")
    u = pick_unit("kw")
    q = pick_qty(random.choice(["kw","dig"]))
    text = f"Muri uyu munsi nabagurishije {p} {u} {q} kuri abakiriya benshi"
    ents = [
        make_entity(text, p, "PRODUCT"),
        make_entity(text, u, "UNIT"),
        make_entity(text, q, "QUANTITY"),
    ]
    return verify_and_clean(text, ents)

def tmpl_en_long():
    """This morning at the market I sold [QUANTITY] [UNIT] of [PRODUCT] to customers"""
    p = pick_product("en")
    q = pick_qty("dig")
    u = pick_unit("en")
    text = f"This morning at the market I sold {q} {u} of {p} to customers"
    ents = [
        make_entity(text, p, "PRODUCT"),
        make_entity(text, q, "QUANTITY"),
        make_entity(text, u, "UNIT"),
    ]
    return verify_and_clean(text, ents)

def tmpl_mixed_kw_prod_en_verb_digit():
    """Sold [PRODUCT_KW] [DIGIT] [UNIT_KW]"""
    p = pick_product("kw")
    q = pick_qty("dig")
    u = pick_unit("kw")
    text = f"Sold {p} {q} {u}"
    ents = [
        make_entity(text, p, "PRODUCT"),
        make_entity(text, q, "QUANTITY"),
        make_entity(text, u, "UNIT"),
    ]
    return verify_and_clean(text, ents)

def tmpl_kw_stock():
    """Nifuza [PRODUCT] [UNIT] [QUANTITY] (I want / ordering)"""
    p = pick_product("kw")
    u = pick_unit(random.choice(["kw","en"]))
    q = pick_qty(random.choice(["kw","dig"]))
    text = f"Nifuza {p} {u} {q}"
    ents = [
        make_entity(text, p, "PRODUCT"),
        make_entity(text, u, "UNIT"),
        make_entity(text, q, "QUANTITY"),
    ]
    return verify_and_clean(text, ents)

# ── Build 200 records ────────────────────────────────────────────────────────

# Weighted template pool — mirrors real-world distribution
TEMPLATE_POOL = (
    [tmpl_kw_sold] * 18 +
    [tmpl_kw_sold_qty_first] * 12 +
    [tmpl_kw_passive] * 8 +
    [tmpl_kw_no_unit] * 6 +
    [tmpl_kw_no_qty] * 4 +
    [tmpl_en_sold] * 14 +
    [tmpl_en_sold_v2] * 8 +
    [tmpl_en_no_unit] * 4 +
    [tmpl_mixed_kw_verb_en_prod] * 16 +
    [tmpl_mixed_en_verb_kw_prod] * 14 +
    [tmpl_mixed_digit_kw] * 12 +
    [tmpl_mixed_digit_en_unit] * 10 +
    [tmpl_mixed_two_products] * 8 +
    [tmpl_mixed_two_products_en] * 4 +
    [tmpl_kw_context] * 6 +
    [tmpl_en_context] * 4 +
    [tmpl_kw_price] * 6 +
    [tmpl_en_price] * 4 +
    [tmpl_kw_question] * 4 +
    [tmpl_en_question] * 4 +
    [tmpl_kw_remaining] * 4 +
    [tmpl_mixed_remaining] * 4 +
    [tmpl_kw_received] * 6 +
    [tmpl_en_received] * 4 +
    [tmpl_kw_ordered] * 4 +
    [tmpl_mixed_ordered] * 4 +
    [tmpl_short_kw] * 6 +
    [tmpl_short_en] * 4 +
    [tmpl_kw_long] * 4 +
    [tmpl_en_long] * 4 +
    [tmpl_mixed_kw_prod_en_verb_digit] * 6 +
    [tmpl_kw_stock] * 6
)

records = []
attempts = 0
while len(records) < 200:
    attempts += 1
    fn = random.choice(TEMPLATE_POOL)
    try:
        r = fn()
        records.append(r)
    except (AssertionError, Exception) as ex:
        pass  # skip bad generations

print(f"Generated {len(records)} records in {attempts} attempts")

# ── Write JSONL ──────────────────────────────────────────────────────────────
# Deliberately named/located so it can never be mistaken for (or accidentally
# copied to) the real ml_experiments/data/annotations.jsonl path.
out_path = str(Path(__file__).parent / "annotations_SYNTHETIC_placeholder.jsonl")
with open(out_path, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Written to {out_path}")
print("Reminder: this is SYNTHETIC data for pipeline testing only — see the")
print("module docstring. It is not the thesis's real annotated test set.")

# ── Quick sanity check ───────────────────────────────────────────────────────
entity_labels = [e["label"] for r in records for e in r["entities"]]
from collections import Counter
print("Entity distribution:", Counter(entity_labels))

missing_unit = sum(1 for r in records if all(e["label"] != "UNIT" for e in r["entities"]))
missing_qty  = sum(1 for r in records if all(e["label"] != "QUANTITY" for e in r["entities"]))
two_products = sum(1 for r in records if sum(1 for e in r["entities"] if e["label"] == "PRODUCT") == 2)
kw_only = sum(1 for r in records if all(c.isascii() or c in 'àâéèêëîïôùûüÿœæ' for c in r["text"]))

print(f"Messages missing UNIT:     {missing_unit}")
print(f"Messages missing QUANTITY: {missing_qty}")
print(f"Messages with 2 products:  {two_products}")
print(f"\nSample messages:")
for r in records[:5]:
    print(" ", r)
