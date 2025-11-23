import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from labels import ID2LABEL, label_is_pii
import os
import re

EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def normalize_stt_email(span_text: str) -> str:
    t = span_text.lower()

    # normalize common STT tokens
    t = t.replace(" dot ", ".")
    t = t.replace(" dot", ".")
    t = t.replace("dot ", ".")
    t = t.replace(" at ", "@")
    t = t.replace(" at", "@")
    t = t.replace("at ", "@")

    # collapse multiple spaces
    t = " ".join(t.split())
    # remove remaining spaces
    t = t.replace(" ", "")
    return t

def is_valid_email(span_text: str) -> bool:
    norm = normalize_stt_email(span_text)
    return EMAIL_REGEX.match(norm) is not None



def is_valid_credit_card(span_text: str) -> bool:
    s = re.sub(r"[\s\-]", "", span_text)
    return s.isdigit() and 13 <= len(s) <= 19  # loose but good enough


def is_valid_phone(span_text: str) -> bool:
    # keep only digits
    s = re.sub(r"[^\d]", "", span_text)
    return 7 <= len(s) <= 15  # allows local + intl



def bio_to_spans(text, offsets, label_ids):
    spans = []
    current_label = None
    current_start = None
    current_end = None

    for (start, end), lid in zip(offsets, label_ids):
        if start == 0 and end == 0:
            continue
        label = ID2LABEL.get(int(lid), "O")
        if label == "O":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None
            continue

        prefix, ent_type = label.split("-", 1)
        if prefix == "B":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
            current_label = ent_type
            current_start = start
            current_end = end
        elif prefix == "I":
            if current_label == ent_type:
                current_end = end
            else:
                if current_label is not None:
                    spans.append((current_start, current_end, current_label))
                current_label = ent_type
                current_start = start
                current_end = end

    if current_label is not None:
        spans.append((current_start, current_end, current_label))

    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir if args.model_name is None else args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()

    results = {}

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            uid = obj["id"]

            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            offsets = enc["offset_mapping"][0].tolist()
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits[0]
                pred_ids = logits.argmax(dim=-1).cpu().tolist()

            
            spans = bio_to_spans(text, offsets, pred_ids)
            ents = []

            for s, e, lab in spans:
                span_text = text[s:e]

                # simple min-length filter to remove junk like 1-char spans
                if (e - s) < 2:
                    continue

                # type-specific validation to improve precision
                if lab == "CREDIT_CARD" and not is_valid_credit_card(span_text):
                    continue
                if lab == "PHONE" and not is_valid_phone(span_text):
                    continue
                if lab == "EMAIL" and not is_valid_email(span_text):
                    continue

                ents.append(
                    {
                        "start": int(s),
                        "end": int(e),
                        "label": lab,
                        "pii": bool(label_is_pii(lab)),
                    }
                )


            results[uid] = ents

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions for {len(results)} utterances to {args.output}")


if __name__ == "__main__":
    main()
