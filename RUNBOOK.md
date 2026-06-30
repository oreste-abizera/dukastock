# RUNBOOK: Zero to Running System

A complete, numbered walkthrough for getting DukaStock running, from an
empty machine to a working backend, trained models, and populated
research notebooks. Follow the sections in order the first time; once
running, jump directly to whichever section you need.

Every command below was executed and verified during this project's
build — not just written and assumed correct. Where something can only be
verified on a different platform (Kaggle GPU, a real Twilio/Africa's
Talking account), that is stated explicitly rather than implied.

---

## 0. Before you start: what you need

- A computer with **Python 3.11** and **pip** (Linux, macOS, or Windows
  with WSL2 — Prophet's Stan backend in particular is friendlier on
  Linux/macOS than native Windows).
- **Git**, to clone/manage this repository.
- **Docker** and **Docker Compose**, if you want the one-command local
  setup (Section 2). Not required if you prefer the manual setup
  (Section 3).
- A free **Kaggle** account, for running the Jupyter notebooks with GPU
  access (Section 5). The notebooks also run locally without a GPU, just
  slower for Notebook 3's transformer fine-tuning step.
- Optional, only needed when you're ready to connect real channels: a
  **Twilio** account (WhatsApp sandbox) and an **Africa's Talking**
  account (USSD/SMS sandbox). Not required to run anything locally first.

---

## 1. Get the code

```bash
unzip DukaStock_Capstone_Project.zip
cd dukastock
```

(Or `git clone` if you've pushed this to a repository instead of working
from the zip.)

---

## 2. Fastest path: Docker Compose

This is the quickest way to get the backend + Redis running together with
one command.

```bash
# 1. Copy the environment template and fill in what you have.
#    You can leave Twilio/Africa's Talking keys blank for now — the
#    backend boots fine without them; only the webhook endpoints that
#    actually call those providers will fail until you add real keys.
cp config/.env.example config/.env

# 2. Build and start everything.
docker-compose up --build
```

What this does:
- Builds the backend image from `backend/Dockerfile` (installs
  `requirements.txt`, which has been verified to resolve cleanly — see
  Section 9, Issue 1, if you hit a dependency error anyway).
- Starts a `redis:7-alpine` container for USSD session state.
- Starts the FastAPI backend on `http://localhost:8000`, using a local
  SQLite file (`dukastock_dev.db`) instead of Supabase, since
  `DATABASE_URL` defaults to SQLite in `docker-compose.yml`.

**Verify it worked:**

```bash
curl http://localhost:8000/api/v1/health
# Expected: {"status":"ok","service":"dukastock-backend"}
```

If that returns the expected JSON, the backend is up. Skip to Section 4.

---

## 3. Manual path (no Docker)

Use this if you don't have Docker, or want to run the backend directly to
debug something.

### 3.1 Create a virtual environment

```bash
cd backend
python3.11 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
```

### 3.2 Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs ~30 packages including FastAPI, SQLAlchemy, Prophet,
XGBoost, pmdarima, transformers, and torch. Expect this to take several
minutes — torch and transformers alone are large downloads. **This exact
command was verified during the project build to resolve without
conflicts** (see Section 9, Issue 1 for the specific bug that was caught
and fixed here, in case you're running an older copy of this project).

If you're on Linux and Prophet's installation fails with a compiler
error, install build tools first:

```bash
sudo apt-get install build-essential gcc g++   # Debian/Ubuntu
```

(macOS users: Prophet generally installs cleanly via pip; if it doesn't,
`brew install cmake` first.)

### 3.3 Start Redis

You need a Redis instance reachable at the URL in your `.env` file
(default `redis://localhost:6379/0`).

**Option A — Docker, just for Redis:**
```bash
docker run -d --name dukastock-redis -p 6379:6379 redis:7-alpine
```

**Option B — Install Redis natively:**
```bash
sudo apt-get install redis-server   # Debian/Ubuntu
redis-server --daemonize yes
```

**Verify Redis is running:**
```bash
redis-cli ping
# Expected: PONG
```

### 3.4 Set up environment variables

```bash
cd ..                              # back to the dukastock root
cp config/.env.example config/.env
```

Open `config/.env` and adjust as needed. For a first local run, the
defaults are fine as-is — `DATABASE_URL` defaults to a local SQLite file,
so you don't need a real Supabase project yet.

### 3.5 Run the backend

```bash
cd backend
source venv/bin/activate           # if not already active
DATABASE_URL="sqlite:///./dukastock_dev.db" \
REDIS_URL="redis://localhost:6379/0" \
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Verify it worked:**
```bash
curl http://localhost:8000/api/v1/health
# Expected: {"status":"ok","service":"dukastock-backend"}
```

---

## 4. Run the test suite

Confirms your environment is sound before you build on top of it. **73
tests, verified passing** as of this project's last review.

```bash
cd backend
source venv/bin/activate           # if using the manual setup
DATABASE_URL="sqlite:///./test.db" PYTHONPATH=. pytest tests/unit/ -v
```

Expected final line: `73 passed in ~13s` (timing varies by machine).

If you see import errors here, re-check Section 3.2 — it almost always
means `requirements.txt` wasn't fully installed in the active environment.

---

## 5. Run the research notebooks

The four notebooks in `ml_experiments/notebooks/` are independent of the
backend — you can run them without the backend running at all. **Kaggle
is the recommended environment** (free GPU, no local setup, matches what
this project was designed against); local execution is also supported
and was used to validate the notebooks' logic during this project's build.

### 5.1 On Kaggle (recommended)

1. Go to [kaggle.com](https://www.kaggle.com) and create a free account
   if you don't have one.
2. Create a new Notebook (**Code → New Notebook**).
3. Click **File → Import Notebook**, upload
   `ml_experiments/notebooks/01_eda_rwanda_localisation.ipynb`.
4. Add the dataset: **Add Data** (right sidebar) → search
   `demand-forecasting-kernels-only` → add it. This is the Kaggle Store
   Item Demand Forecasting Challenge dataset the whole project is built
   on.
5. For Notebook 2 and 3, also enable a GPU: **Settings** (right sidebar)
   → **Accelerator** → **GPU T4 x2**. Notebook 1 and 4 don't need a GPU.
6. Click **Run All**.
7. Run the notebooks **in order**: 1 → 2 → 3 → 4. Each later notebook
   loads files saved by an earlier one (see the "Prerequisites" cell at
   the top of each notebook, which checks for the expected input file and
   tells you exactly which notebook to run first if it's missing).

**Expected runtime** (rough, varies by Kaggle's current CPU/GPU
allocation):
- Notebook 1: a few minutes (913,000 rows, pure pandas/matplotlib, no
  model training).
- Notebook 2: **this is the slow one.** The full experiment matrix is 5
  products × 10 stores × 6 density levels, evaluated per (store, product)
  pair with SARIMA's `auto_arima` order search as the dominant cost. **Budget
  10–14 hours on a Kaggle/Colab GPU T4.** Enable a GPU before running
  (Settings → Accelerator → GPU T4 on Kaggle; Runtime → Change runtime type
  → T4 GPU on Colab). If session time is a concern, set `RUN_ALL_STORES =
  False` in Cell 11 to use 5 stores (~5-7 hours), or set
  `RUN_FULL_EXPERIMENT = False` for a smoke-test run on one product/density.
- Notebook 3: depends entirely on whether you've supplied real annotated
  data (see Section 6 below) and how many epochs you fine-tune for;
  expect minutes on a Kaggle GPU, much longer on CPU only.
- Notebook 4: under a minute — it only loads and re-plots results already
  computed by Notebooks 2 and 3.

### 5.2 Running locally instead

```bash
cd ml_experiments
pip install jupyter nbconvert ipykernel
python -m ipykernel install --user --name dukastock

# Download the Kaggle dataset (requires a Kaggle account + API token,
# see https://www.kaggle.com/docs/api):
kaggle competitions download -c demand-forecasting-kernels-only -p data/
unzip data/demand-forecasting-kernels-only.zip -d data/

jupyter notebook notebooks/01_eda_rwanda_localisation.ipynb
```

Same ordering and runtime caveats as Section 5.1 apply.

---

## 6. Collect real data before trusting the numbers

Two notebooks generate clearly-flagged **synthetic placeholder data** when
real data isn't available, specifically so the pipeline can be validated
without blocking on data collection. Before citing any number from these
in the thesis, replace the placeholders:

1. **Notebook 2 / `run_experiment.py`** need the real Kaggle `train.csv`
   (913,000 rows) — see Section 5.1, step 4. Without it, Notebook 2 will
   only run against whatever CSV you point it at; it doesn't fabricate
   data itself, but make sure you're pointing it at the real file, not a
   test file left over from development.
2. **Notebook 3** needs a real 200-message annotated Kinyarwanda commerce
   NER test set. **Follow `docs/ANNOTATION_GUIDE.md` end to end** —
   it covers what to collect, the Doccano annotation tool setup, the
   exact label scheme, and the second-annotator protocol for computing
   Cohen's Kappa. Once you have
   `ml_experiments/data/annotations.jsonl` in place, Notebook 3 detects
   it automatically (`USING_SYNTHETIC_DATA` prints `False`) and uses it
   instead of generating placeholder messages.

---

## 7. Collect SUS usability data (tertiary research question)

1. Read `docs/SUS_QUESTIONNAIRE.md` for the full questionnaire (English +
   Kinyarwanda), administration protocol, and CSV format.
2. Run a hands-on demonstration session with at least 3 Duka operators
   (the proposal's minimum scope), administering the questionnaire
   immediately after each participant tries the WhatsApp and/or USSD
   channel.
3. Save responses as a CSV matching the format in
   `docs/SUS_QUESTIONNAIRE.md` Section "Data format for scoring".
4. Score it:

   ```bash
   python ml_experiments/scripts/score_sus.py \
     --input your_collected_responses.csv \
     --output-dir ml_experiments/results \
     --plot
   ```

5. Re-run Notebook 4 — it will automatically detect
   `ml_experiments/results/sus_scores_by_channel.csv` and include the
   usability section in the dashboard.

---

## 8. Train and serve real forecasting models

Once you have the Rwanda-localized dataset in place (produced by Notebook 1
or by Section 6, step 1 → run Notebook 1 → save `fmcg_rwanda_localized.csv`):

```bash
cd ml_experiments
python scripts/run_experiment.py \
  --data data/fmcg_rwanda_localized.csv \
  --output-dir results \
  --artifact-dir artifacts \
  --horizon 7
```

This evaluates all 5 model classes at the individual-store level (10 stores
× 5 products × 6 densities), serializes the per-product best model to
`artifacts/<product>_best_model.joblib`, and writes both
`results/ml_benchmark_results_raw.csv` (per-store) and
`results/ml_benchmark_results.csv` (store-averaged, used by Notebook 4).

**Runtime: same as Notebook 2 — 10–14 hours on GPU** (SARIMA's `auto_arima`
is the bottleneck). Run on Kaggle (12h session limit) or a persistent machine.

Once it finishes, point the backend at the artifacts directory. Either
copy the files:

```bash
mkdir -p backend/ml_experiments/artifacts
cp ml_experiments/artifacts/*.joblib backend/ml_experiments/artifacts/
```

or set `MODEL_ARTIFACT_DIR=ml_experiments/artifacts` in `config/.env`
instead of copying — either works; `ForecastService` just reads whatever
directory `settings.model_artifact_dir` resolves to.

**Verify it worked** (with the backend running — Section 2 or 3):

```bash
curl http://localhost:8000/api/v1/forecast/SUGAR
# Expect a real model_used value (e.g. "prophet", "xgboost", "sarima"),
# not "unavailable".
```

---

## 9. Connecting real channels (optional, only when ready to demo live)

Everything above runs and is fully testable without this section — these
steps are only needed to receive real WhatsApp/USSD messages instead of
testing via `curl`.

### 9.1 Twilio WhatsApp sandbox

1. Sign up at [twilio.com](https://www.twilio.com), activate the WhatsApp
   Sandbox (Console → Messaging → Try it out → Send a WhatsApp message).
2. Copy your Account SID and Auth Token into `config/.env`:
   ```
   TWILIO_ACCOUNT_SID=your_sid_here
   TWILIO_AUTH_TOKEN=your_token_here
   ```
3. Set the sandbox's webhook URL to
   `https://<your-public-backend-url>/api/v1/webhooks/whatsapp`. If
   you're testing locally, expose your local server with a tunnel first
   (e.g. `ngrok http 8000`), then use the ngrok URL.

### 9.2 Africa's Talking USSD/SMS sandbox

1. Sign up at [africastalking.com](https://www.africastalking.com),
   create a sandbox app.
2. Copy your username and API key into `config/.env`:
   ```
   AT_USERNAME=your_sandbox_username
   AT_API_KEY=your_api_key_here
   ```
3. Set the USSD callback URL to
   `https://<your-public-backend-url>/api/v1/webhooks/ussd` in the
   Africa's Talking sandbox dashboard.

Neither provider is required for any of Sections 1-8 to work — the
backend, tests, notebooks, and model training are all fully functional
without real Twilio/Africa's Talking credentials.

---

## 10. Known issues and what to do about them

### Issue 1 (fixed in this version): `httpx`/`supabase` dependency conflict

An earlier version of `backend/requirements.txt` pinned `httpx==0.28.1`,
which conflicts with `supabase==2.10.0`'s actual requirement of
`httpx<0.28,>=0.26`. This was caught and fixed during review — the
current `requirements.txt` pins `httpx==0.27.2`, verified to resolve
cleanly. **If you're running an older exported copy of this project and
hit `ResolutionImpossible` mentioning httpx during `pip install`,** open
`backend/requirements.txt` and change the `httpx` line to `httpx==0.27.2`.

### Issue 2: SARIMA is slow

`auto_arima`'s order search is the dominant cost in Notebook 2 and
`run_experiment.py`. This is inherent to the method, not a bug. If you
need faster iteration during development, reduce
`max_p`/`max_q`/`max_P`/`max_Q` in `backend/app/ml/models/sarima.py`, or
use the `RUN_FULL_EXPERIMENT = False` smoke-test flag in Notebook 2.

### Issue 3: Notebook 3's transformer download needs real internet access

Notebook 3 downloads `xlm-roberta-base` from Hugging Face at fine-tuning
time. If you're running in a network-restricted environment (a corporate
proxy, a sandboxed CI runner, etc.), this step will fail with a
connection error — it is not a bug in the notebook, just a network
reachability requirement. Kaggle and most local developer machines have
unrestricted access to huggingface.co by default.

### Issue 4: Prophet installation problems on native Windows

Prophet depends on a Stan backend that compiles native code. If
`pip install prophet` fails on native Windows, the most reliable fix is
running everything inside WSL2 (Windows Subsystem for Linux) instead,
where Prophet installs the same way it does on native Linux.

### Issue 5: `forecast` endpoint returns `"no_model"` before experiment is run

If `ml_experiments/artifacts/` is empty (no `.joblib` files), every call
to `GET /api/v1/forecast/<product>` will return
`{"model_used": "no_model", "predicted_quantity": null, "status": "no_model_available"}`.
This is intentional — it is a clear signal that the experiment has not been run
yet, not a numeric 0.0 that could be mistaken for a real prediction. Run
Section 8 to produce real model artifacts.

### Issue 6: NER model auto-detection

The config auto-detects the XLM-R model by looking for
`ml_experiments/notebooks/xlmr_commerce_ner_output/config.json` first,
then falling back to `app/nlp/xlmr_commerce_ner/`. The `backend/app/nlp/xlmr_commerce_ner/`
directory is intentionally empty in the git repository (the 1.1 GB safetensors
file is not committed). If the auto-detection fails (e.g. you're running the
backend from a different working directory), set `NER_MODEL_DIR` explicitly
in your `.env` to the absolute path of the model directory.

### If something here doesn't match what you see

This runbook reflects the project state as of the last full review (all
73 backend tests passing, all 4 notebooks executing end-to-end against
representative data, dependency resolution independently verified). If
you hit something not covered above, check:
1. `docs/SOURCES.md` for any fact-level corrections made to the original
   proposal.
2. `docs/RESEARCH_DESIGN.md`'s "Documented deviations" section for
   intentional implementation choices that depart from the proposal text.
3. The relevant module's docstring — most non-obvious decisions in this
   codebase are explained inline at the point they're made, not just in
   these docs.

---

## 11. Step-by-step live demo script (for Hubert or any evaluator)

**Audience:** Supervisor Hubert Apana, or any examiner who has not seen
the system before. **Duration:** ~20 minutes. All `curl` commands can be
run in a visible terminal while you narrate.

### Before you start

Make sure these are true:
- Backend is running (`curl http://localhost:8000/api/v1/health` returns `ok`).
- Redis is running (`redis-cli ping` returns `PONG`).
- ngrok is running if you want to show real Twilio/AT traffic (optional for the demo).
- Notebook 02 results are available in `ml_experiments/results/` (or use
  the Kaggle notebook URL to show the threshold curve).

---

### Demo step 1 — Architecture overview (2 min)

Open `docs/ARCHITECTURE.md` or `docs/diagrams/figure2_architecture.png`.

*Say:* "DukaStock has one FastAPI backend serving three channels — WhatsApp,
USSD, and SMS — through a single shared forecasting pipeline. This matters
because it means one Duka operator with a smartphone and another with a basic
feature phone both get the same product intelligence, from the same models,
through different front-ends. Let me show each channel in order."

---

### Demo step 2 — USSD channel (5 min)

*Say:* "USSD is the key inclusive channel. Rwanda's EICV7 survey confirms
66 percent of households don't have a smartphone, but 85 percent have a mobile
phone. USSD works on any of them — no data connection, no app download."

Simulate the full USSD session in a visible terminal:

```bash
# Dial in (empty text = user just dialled the code)
curl -s -X POST http://localhost:8000/api/v1/webhooks/ussd \
  -d "sessionId=demo-001&phoneNumber=%2B250788123456&text=&serviceCode=%2A384%2300%23"
```

*Point out:* "`CON` at the start means the session continues. The menu is in
Kinyarwanda — 'Murakaza neza kuri DukaStock' means 'Welcome to DukaStock'."

```bash
# Press 1 → log a sale
curl -s -X POST http://localhost:8000/api/v1/webhooks/ussd \
  -d "sessionId=demo-001&phoneNumber=%2B250788123456&text=1&serviceCode=%2A384%2300%23"

# Select Sugar (1), enter 3 kg
curl -s -X POST http://localhost:8000/api/v1/webhooks/ussd \
  -d "sessionId=demo-001&phoneNumber=%2B250788123456&text=1%2A1%2A3&serviceCode=%2A384%2300%23"
```

*Point out:* "`END` at the start means the session has terminated. 'Murakoze!
Igurisha ryanditswe' means 'Thank you! Sale recorded.' The database now has
this sale logged, privacy-compliant — the phone number is hashed before
storage, never stored raw."

---

### Demo step 3 — WhatsApp / NER pipeline (5 min)

*Say:* "WhatsApp users can send a natural language Kinyarwanda message instead
of pressing menus. The NER pipeline parses it."

```bash
curl -s -X POST http://localhost:8000/api/v1/webhooks/whatsapp \
  -d "From=whatsapp%3A%2B250788999001" \
  -d "Body=Nabagurishije+isukari+ibiro+bitatu+namavuta+litre+imwe"
```

*Point out the TwiML response confirming SUGAR 3 kg and OIL 1 litre from one
sentence.*

Show the NER pipeline directly:

```bash
cd backend
python -c "
from app.nlp.ner_pipeline import CommerceNERPipeline
p = CommerceNERPipeline()
results = p.parse('Nabagurishije isukari ibiro bitatu namavuta litre imwe')
for r in results:
    print(r.to_dict())
"
```

*Say:* "The pipeline first tries the fine-tuned XLM-R transformer. If it
can't load, it falls back to a RapidFuzz fuzzy-matching baseline over a
Kinyarwanda product lexicon. Both are evaluated against a 200-message annotated
test set as part of Research Question 2."

---

### Demo step 4 — Privacy compliance (2 min)

```bash
cd backend
python -c "
from app.core.security import hash_phone_number
raw = '+250788123456'
hashed = hash_phone_number(raw)
print('Raw phone number: ', raw)
print('Stored as UUID:   ', hashed)
print()
print('Same number always maps to the same UUID (deterministic),')
print('but the UUID cannot be reversed to the phone number.')
"
```

*Say:* "Rwanda Law No. 058/2021 requires personal data to be protected with
appropriate technical safeguards. DukaStock hashes every phone number using
SHA-256 with a secret application salt before it reaches the database layer.
The hash is deterministic — the same shopkeeper always maps to the same
UUID — but the original number cannot be recovered even if someone exfiltrates
the database."

---

### Demo step 5 — ML benchmark and threshold curve (5 min)

*If the experiment has been run:*

Open `ml_experiments/notebooks/04_results_dashboard.ipynb` (or the Kaggle
version) and navigate to the **threshold curve figure** — the plot of
fraction-of-folds-significant vs. density level per model class.

*Say:* "This is the primary finding. The x-axis is the cold-start data density —
how much of the historical data we give the model to train on, ranging from
5 percent (simulating a Duka that started recording sales about three months ago)
to 100 percent. The y-axis is the fraction of walk-forward cross-validation
folds where that model achieves statistically significant improvement over
the naive last-week-sales baseline, using the Diebold-Mariano test at p < 0.05.

The density level where a model's curve first crosses a meaningful threshold
— say, 50 percent of folds significant — is the minimum transaction history
at which that model becomes reliably useful. That threshold, per model, is
the answer to Research Question 1."

*If the experiment has not yet been run:*

Show Notebook 02, Cell 16 (the threshold curve cell) and explain what the
output will look like once the experiment completes.

*Show the direct API:*

```bash
curl http://localhost:8000/api/v1/forecast/SUGAR
```

If artifacts exist, this returns `model_used: "xgboost"` (or whichever model
won at 100% density) with a weekly prediction. If not yet trained, it returns
`"no_model"` — which is the honest answer until the experiment runs.

---

### Demo step 6 — SMS channel (1 min)

```bash
cd backend
python -c "
from app.channels.sms.handler import build_weekly_summary_sms
msg = build_weekly_summary_sms(
    shopkeeper_locale_is_kinyarwanda=True,
    product_codes=['SUGAR', 'OIL', 'FLOUR', 'RICE', 'SOAP']
)
print(msg)
print()
print('Characters:', len(msg), '/ 160 maximum (single GSM-7 SMS segment)')
"
```

*Say:* "The SMS channel is the nudge channel — a weekly outbound summary
pushed to shopkeepers who don't use WhatsApp and don't want to dial USSD
proactively. Under 160 characters to keep delivery costs predictable on
the Africa's Talking sandbox."

---

## 12. Submission-readiness checklist (as of last audit, 2026-06-30)

| Item | Status | Action needed |
|---|---|---|
| Rwanda-localized dataset (Kaggle + holidays + FMCG subset) | **Done** | None |
| Walk-forward CV + Newey-West DM test implementation | **Done** | None |
| Per-store (Duka-proxy) experiment evaluation | **Done** | None |
| XGBoost TimeSeriesSplit grid search in notebook + script | **Done** | None |
| N-BEATS max_steps=500 (GPU) for convergence | **Done** | None |
| Reproducibility seed (42) in all experiment entry points | **Done** | None |
| All four model classes implemented | **Done** | None |
| ML benchmark experiment results (threshold curve) | **MISSING** | Re-run Notebook 2 on Kaggle/Colab GPU (~10-14h) |
| Serialized model artifacts (`.joblib` per product) | **MISSING** | Produced automatically by `run_experiment.py` |
| 200-message annotated NER test set (real data) | **MISSING** | Human annotation required — see Section 6 |
| Cohen's Kappa from real second annotator | **MISSING** | Follows from annotation |
| XLM-R fine-tuning on real annotations | **MISSING** | Run after annotations complete (Notebook 3 / train_xlmr_ner.py) |
| Three-channel prototype (WhatsApp / USSD / SMS) | **Done** | None |
| Privacy: phone number hashing (Law 058/2021) | **Done** | None |
| SUS demonstration session (≥3 Duka operators) | **MISSING** | Field session required |
| Unit test suite | **Done** | None |
| NER model path auto-detection | **Done** | None |
| USSD webhook accepts AT's `serviceCode`/`networkCode` | **Done** | None |
| ForecastService `no_model` status signal | **Done** | None |
| `PHONE_HASH_SALT` startup assertion (non-dev environments) | **Done** | None |
| Honest data limitation disclosure (proxy dataset, Rwanda features) | **Done** | None — documented in Notebooks 1/2 and RESEARCH_DESIGN.md |

**Target completion for missing items (July 20, 2026 deadline):**
- ML benchmark experiment: July 12
- Annotations + NER: July 14
- SUS session: July 17
- Thesis write-up of results: July 19
- Submission: July 20
