# GPU ingest on a rented EC2 box (overnight runbook)

A solo, throwaway recipe: rent one AWS GPU instance, throw a large corpus at
`bartleby scribe` overnight, copy the finished `.db` off, kill the box. This is
the single-user, maximize-throughput sibling of the shared/persistent box in
[#158](https://github.com/jswest/bartleby/issues/158) — different goals, don't
conflate them.

The unlock that makes a GPU worth renting is the **`docling_device`** config
knob (added in #159). Docling defaults to CPU because Apple Silicon's MPS
backend crashes its vision models; on a Linux/CUDA box that failure mode doesn't
exist, so `docling_device: cuda` moves docling's layout/OCR/TableFormer models
onto the card.

---

## What runs where

Ingestion has three model workloads. On a CUDA box, all three can use the GPU:

| Workload | What | How it reaches the GPU |
|---|---|---|
| **Docling** | PDF/HTML/MD layout, OCR, TableFormer | `docling_device: cuda` (this is the new knob) |
| **Embeddings** | `BAAI/bge-base-en-v1.5`, 768-dim | Auto — `sentence-transformers` detects CUDA when torch has it |
| **LLM** (summaries + image analysis) | local `qwen3-vl:30b`, or a cloud model | Local ollama auto-detects CUDA; cloud needs no GPU |

Docling and embeddings run **in-process** with `bartleby scribe`, so the CLI
itself must run on the box. The LLM can be local (ollama on the same box) or a
cloud API — that choice drives the instance sizing and the cost (see below).

---

## Pick an instance

The constraint is VRAM, and the hog is the local LLM. `qwen3-vl:30b` is
**Qwen3-VL-30B-A3B** (a 30B-total / ~3B-active MoE), an ~20 GB download at Q4;
loaded with KV cache and the vision encoder it wants **~20–22 GB resident**.
Docling adds ~1–2 GB, embeddings ~0.5 GB — all on the same card.

| Instance | GPU | VRAM | On-demand* | Spot* | Fits local `qwen3-vl:30b`? |
|---|---|---|---|---|---|
| `g5.xlarge` | A10G | 24 GB | $1.006/hr | $0.476/hr | **Tight** — 30B + docling + embeddings will fight for 24 GB |
| `g6e.xlarge` | L40S | 48 GB | $1.861/hr | $1.123/hr | **Comfortable** |

\* us-east-1, Linux, mid-2026. **Verify current pricing** — rates and spot
availability move. (`instances.vantage.sh`, AWS console.)

Rules of thumb:
- **All-local LLM** → `g6e.xlarge` (48 GB). Don't try to cram the 30B onto a
  24 GB A10G alongside docling+embeddings; you'll OOM or thrash.
- **Cloud LLM** (e.g. gpt-5-nano) → the 30B is gone from VRAM, so `g5.xlarge`
  (24 GB) comfortably holds just docling+embeddings. Cheaper card.
- Use a **CUDA-ready AMI** (AWS Deep Learning Base GPU AMI, Ubuntu) so the
  NVIDIA driver + CUDA toolkit are preinstalled. Confirm with `nvidia-smi`.

---

## Bootstrap

```bash
# 0. Confirm the GPU is visible
nvidia-smi

# 1. uv + bartleby (with docling extras)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install --with docling bartleby     # or: uv pip install 'bartleby[docling]'

# 2a. ALL-LOCAL LLM: install ollama + pull the model (~20 GB download)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3-vl:30b

# 3. Configure. The wizard now asks for the Docling device when docling is the
#    pdf/html converter — answer "cuda".
bartleby config
#   provider:        ollama  (all-local)   OR   openai  (cloud, see 2b)
#   model:           qwen3-vl:30b           OR   gpt-5-nano
#   pdf_converter:   docling
#   Docling device:  cuda      <-- the knob
#   (vision_provider likewise: ollama qwen3-vl:30b, or openai gpt-5-nano)

# 2b. CLOUD LLM instead of ollama: skip step 2a, set provider openai in the
#     wizard, and export the key (gpt-5-nano is multimodal, so it covers both
#     the summary and vision roles):
export OPENAI_API_KEY=sk-...

# 4. Create the project
bartleby project create mycorpus
bartleby project use mycorpus
```

`docling_device: cuda` lives in `~/.bartleby/config.yaml`. There's no
`--docling-device` CLI flag — it's config-only by design; set it once.

---

## Get the corpus onto the box

15 GB over the network: stage through S3 rather than a long scp.

```bash
# from your machine
aws s3 sync ./corpus s3://my-bucket/corpus/
# on the box
aws s3 sync s3://my-bucket/corpus/ ~/corpus/
```

---

## Smoke test FIRST — don't turn it loose blind

Docling-on-CUDA is a fresh path; prove it before committing to a multi-hour run.

```bash
# Ingest ~100 mixed files
mkdir ~/smoke && cp $(ls ~/corpus/* | head -100) ~/smoke/   # rough sample
time bartleby scribe --files ~/smoke --project mycorpus

# In another shell, while it runs, confirm work is on the GPU:
watch -n1 nvidia-smi
#   - the bartleby python process should show GPU memory + util  (docling on CUDA)
#   - ollama (all-local) should also appear
```

Two things to take from the smoke test:
1. **Verify docling is actually on the GPU** — a python process in `nvidia-smi`.
   If it's CPU-pinned, recheck `docling_device: cuda` in the config.
2. **Measure the per-file rate** (`time` ÷ 100). Multiply by 9,000 for a real
   wall-clock estimate — this is the only honest way to predict the run length
   and therefore the GPU-hours cost.

---

## Run it overnight

Detach so an SSH drop doesn't kill a multi-hour job:

```bash
tmux new -s ingest
bartleby scribe --files ~/corpus --project mycorpus 2>&1 | tee ~/ingest.log
#   Ctrl-b d to detach;  tmux attach -t ingest to return
```

**Resumability caveat:** confirm from the smoke test what `bartleby scribe` does
on re-run over already-ingested files (skip vs duplicate vs error) before you
trust an unattended overnight job — especially on spot, where the box can vanish
mid-run. If re-running isn't clean, prefer **on-demand** over spot for the long
haul, and keep the `tee` log so you can see where it stopped.

**If you OOM** (watch the log / `nvidia-smi`):
- Drop to a smaller LLM, or move the LLM to a cloud provider (frees ~20 GB).
- `summary_depth: none` skips per-doc summaries entirely if you only want
  retrieval — removes the LLM from the summary path.
- Lower `vision_max_dimension` to shrink image tokens.

---

## Teardown

The `.db` is self-sufficient for research (search/read/cite all come from the
DB; raw docs are only needed to re-ingest or to serve images). Copy it off, then
kill the box.

```bash
aws s3 cp ~/.bartleby/projects/mycorpus/*.db s3://my-bucket/done/   # when idle
# then terminate the instance (and any spot request)
```

---

## What will this cost? (local GPU vs. gpt-5-nano)

Two ways to pay for the LLM work, in the **same ~$15–30 ballpark** for one
overnight run on this corpus. Numbers are worked examples with stated
assumptions — **plug in your own measured rates.**

**Assumptions (adjust to your corpus):** ~9,000 docs, one summary each. Average
summarization input ≈ 12,000 tokens/doc (many short docs, some long; the cap is
`max_summarize_tokens`, default 50,000), output ≈ 600 tokens/doc. Vision/image
analysis is *additive* and scales with how many images your corpus has — not
included below.

### Option A — all-local (ollama on the GPU)

You pay **only for GPU-hours**; no API bill. Needs the 48 GB box to hold the 30B.

```
g6e.xlarge, ~12 h overnight:
  on-demand  12 × $1.861 ≈ $22
  spot       12 × $1.123 ≈ $13
```

### Option B — gpt-5-nano (cloud LLM) + a cheaper box

The 30B leaves VRAM, so docling+embeddings fit a 24 GB `g5.xlarge`. You pay a
**small API bill + cheaper GPU-hours.** gpt-5-nano: **$0.05 / 1M input**,
**$0.40 / 1M output** (multimodal, so it also does vision).

```
API (summaries):
  input   9,000 × 12,000 =  108M tok × $0.05/M ≈ $5.40
  output  9,000 ×    600 =  5.4M tok × $0.40/M ≈ $2.16
  ------------------------------------------------ ≈ $7.6   (≈ $8–25 across the
                                                             doc-length range;
                                                             ~$25 if most docs
                                                             hit the 50k cap)
GPU box (docling+embeddings only), g5.xlarge ~12 h:
  on-demand  12 × $1.006 ≈ $12
  spot       12 × $0.476 ≈ $6
  ------------------------------------------------
Option B total ≈ $14–32
```

### So which?

Cost is close to a wash. Decide on the **non-cost** factors:
- **Privacy** — all-local keeps every document on your box; gpt-5-nano sends
  document text (and images) to OpenAI.
- **Image-heavy corpus** — vision tokens push the API bill up (Option B) but are
  "free" GPU time locally (Option A). If you have lots of images, local often
  wins on cost too.
- **Spot capacity** — if you can reliably get spot, all-local on `g6e` spot
  (~$13) is both cheap and private.
- **Wall clock** — measure it from the smoke test; a slower run shifts both the
  GPU-hours and (for local) nothing on the API side. The pricing above assumes
  ~12 h; your number may differ.

Pricing snapshot: mid-2026, us-east-1. Re-check before you rely on it.
