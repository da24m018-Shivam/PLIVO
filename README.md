# PII NER – STT-Based Synthetic Data + Token Classification Model

This repository contains my complete solution for the **IIT Madras – PII NER Assignment (2025)**.

The goal is to build a lightweight, low-latency Named-Entity Recognizer for detecting PII entities from noisy, STT-style (speech-to-text) utterances, using:

- Synthetic data generation
- Token classification models (HuggingFace Transformers)
- BIO tagging + span evaluation
- Latency measurement (batch_size = 1)
- Robust post-processing for PII precision

---

## 📁 Repository Structure

```
.
├── data/
│   ├── train.jsonl                # synthetic STT-style train dataset
│   ├── dev.jsonl                  # synthetic STT-style dev dataset
│   └── test.jsonl                 # original assignment test set
│
├── out/                           # saved model + predictions + metrics
│   ├── dev_pred.json
│   └── test_pred.json
│
└── src/
    ├── dataset.py                 # dataset → tokenization → BIO labels
    ├── labels.py                  # label ↔ id mappings (BIO scheme)
    ├── model.py                   # RoBERTa token classification model
    ├── train.py                   # training loop + optimizer + scheduler
    ├── predict.py                 # inference + BIO → span decoding
    ├── eval_span_f1.py            # span-level F1 (per-entity + PII)
    ├── measure_latency.py         # p50 / p95 latency measurement
    ├── generate_synthetic_data.py # STT-style synthetic data generator
    └── validate_data.py           # integrity checker for synthetic dataset
```

---

## 🚀 Approach Summary

### 1️⃣ Synthetic Data Generation (STT-focused)

Since the assignment requires speech-style PII, I built a custom generator:

✓ Indian names, cities, locations  
✓ Mixed email formats ("dot", "at", no punctuation, spelled-out)  
✓ Phone numbers in multiple formats (digits, spelled-out, +91, grouped)  
✓ Realistic STT noise:
- "g male", "dilli", "bumbai", "varma/sharmma"
- fillers ("uh", "umm", "like", "yaar")
- light typos
- homophones ("too/to", "one/won")

✓ Noise added only to non-PII spans (entity text is preserved)  
✓ Spans recalculated after noise injection

This ensures realistic ASR noise without destroying label alignment.

**Generate synthetic data:**

```bash
python src/generate_synthetic_data.py --gen-train --gen-dev
```

---

### 2️⃣ Model Choice

After testing DistilBERT, BERT-base, and RoBERTa-base, the final selected model is:

⭐ **roberta-large**

- Highest PII F1 among tested models
- Latency still meets assignment constraints
- Robust BPE tokenizer handles noisy STT text well
- Excellent performance on EMAIL, CREDIT_CARD, PHONE

---

### 3️⃣ Training

Runs a standard token classification fine-tuning loop:

- Batch size: 32
- Epochs: 13
- LR: 2e-5
- Max length: 128
- Warmup + linear scheduler
- Gradient clipping (1.0)

**Train command:**

```bash
python src/train.py --model_name roberta-large --train data/train.jsonl --dev data/dev.jsonl --out_dir out
```

The fine-tuned model + tokenizer are saved into `out/`.

---

### 4️⃣ Inference + Robust PII Post-Processing

After BIO decoding, extra steps improve PII precision:

- Remove 1-character spans
- Validate CREDIT_CARD by numeric length (13–19 digits)
- Validate PHONE (7–15 digits)
- Normalize STT emails:
  - `ramesh dot sharma at gmail dot com` → `ramesh.sharma@gmail.com`
- Validate EMAIL with regex
- Keep CITY, LOCATION, PERSON_NAME as-is
- Mark all PII spans with `"pii": true`

**Inference command:**

```bash
python src/predict.py --model_dir out --input data/dev.jsonl --output out/dev_pred.json
```

---

### 5️⃣ Evaluation (Span F1)

Span evaluator checks exact start/end + label match.

Reports:
- Per-entity F1
- PII-only precision/recall/F1 (main metric)
- Non-PII metrics
- Macro F1

**Run evaluation:**

```bash
python src/eval_span_f1.py --gold data/dev.jsonl --pred out/dev_pred.json
```

---

### 6️⃣ Latency Measurement

Latency run uses:
- batch_size = 1
- 50 runs
- forward pass + tokenization latency
- p50 and p95 reported

**Command:**

```bash
python src/measure_latency.py --model_dir out --input data/dev.jsonl --runs 50
```

---

## 📊 Final Metrics (Reported in Loom)

You will include:

- Per-entity F1
- PII Precision
- PII Recall
- PII F1 (main score)
- Overall Macro F1
- Latency p50 / p95 (ms)

These come directly from `eval_span_f1.py` and `measure_latency.py`.

---

## 🧪 Test Predictions

Generate predictions for the assignment test set:

```bash
python src/predict.py --model_dir out --input data/test.jsonl --output out/test_pred.json
```

`out/test_pred.json` is included as part of the final submission.

---

## 🎥 Loom Video Checklist

Your Loom should cover:

- ✔ Final results + metrics
- ✔ Codebase walkthrough (src/* overview)
- ✔ Synthetic data generation logic
- ✔ Model & tokenizer selection
- ✔ Key hyperparameters
- ✔ PII precision/recall/F1 explanation
- ✔ Latency trade-offs (p50/p95)

---

## 📌 How to Reproduce Entire Pipeline

```bash
# 1. Generate synthetic data
python src/generate_synthetic_data.py --gen-train --gen-dev

# 2. Train RoBERTa-large
python src/train.py --model_name roberta-base --train data/train.jsonl --dev data/dev.jsonl --out_dir out

# 3. Predict on dev
python src/predict.py --model_dir out --input data/dev.jsonl --output out/dev_pred.json

# 4. Evaluate span F1
python src/eval_span_f1.py --gold data/dev.jsonl --pred out/dev_pred.json

# 5. Measure latency
python src/measure_latency.py --model_dir out --input data/dev.jsonl --runs 50

# 6. Predict on test
python src/predict.py --model_dir out --input data/test.jsonl --output out/test_pred.json
```
