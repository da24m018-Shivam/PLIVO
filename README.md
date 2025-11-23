# PII NER – STT-Based Synthetic Data + Token Classification Model
Code is in master branch
Output : https://drive.google.com/drive/folders/1E0ET__XSReXianNGXr9llehm_mYSHCOS?usp=drive_link

Data file : https://drive.google.com/file/d/17Ob_O2GL03-SQ0JHV1LxByBJv0ldTCGc/view?usp=sharing

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


```
