import json
import random
import argparse
import string
from typing import List, Dict

# ---------- basic vocab ----------

FIRST_NAMES = [
    # existing
    "ramesh", "priyanka", "rohan", "megha", "suresh",
    "anita", "vikram", "kiran", "akash", "neha",
    # more common Indian names
    "ajay", "vijay", "rahul", "sanjay", "pankaj",
    "deepak", "amit", "manish", "arjun", "rohit",
    "pooja", "swati", "kavita", "shruti", "nisha",
    "sneha", "sonali", "alka", "radhika", "tanya",
]

LAST_NAMES = [
    # existing
    "sharma", "verma", "mehta", "iyer", "rao",
    "gupta", "singh", "patel", "nair", "das",
    # more
    "reddy", "kulkarni", "joshi", "chatterjee", "banerjee",
    "mishra", "tiwari", "agrawal", "kapoor", "bose",
]

CITIES = [
    # existing
    "mumbai", "delhi", "chennai", "bangalore", "kolkata",
    "hyderabad", "pune", "ahmedabad", "jaipur", "kochi",
    # more tier-1 / tier-2
    "noida", "gurgaon", "thane", "surat", "lucknow",
    "indore", "bhopal", "nagpur", "visakhapatnam", "ludhiana",
    "vadodara", "patna", "ghaziabad", "agra", "nashik",
]

LOCATIONS = [
    "andheri east", "andheri west", "bandra kurla", "mg road", "whitefield",
    "velachery", "salt lake", "bopal area", "banjara hills", "hinjewadi",
    "hsr layout", "indiranagar", "sector 18", "sector 62", "old airport road",
]

EMAIL_PROVIDERS = [
    "gmail", "yahoo", "outlook", "hotmail", "rediffmail",
    "icloud", "protonmail",
]

MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]

NUM_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}

# ---------- NOISE DICTIONARIES ----------

ASR_NOISE = {
    "gmail": ["g male", "g mail", "gmal", "gmail", "gml"],
    "yahoo": ["y hoo", "yahu", "yaho", "yahoo"],
    "outlook": ["out look", "outlok", "ootlook", "outlook"],
    "hotmail": ["hot mail", "hotmale", "hotmal"],
    "rediffmail": ["redif mail", "redif", "rediff mail"],
    "mumbai": ["bumbai", "mumbay", "mumba"],
    "delhi": ["dilli", "delii", "deli"],
    "bangalore": ["bengaluru", "bangluru", "bangalore"],
    "kolkata": ["calcata", "kolkatta", "kolkatta"],
    "verma": ["varma", "vermaa"],
    "sharma": ["sarma", "sharmma"],
    "dot": ["daat", "daught", "dot", "dhot"],
    "at": ["aat", "et", "ad"],
}

FILLERS = [
    "uh", "umm", "hmm", "like", "basically",
    "actually", "you know", "i mean", "sort of",
    "kind of", "so", "yaar",
]

HOMOPHONES = {
    "two": ["to", "too", "tu"],
    "four": ["for", "fore"],
    "one": ["won"],
    "eight": ["ate"],
    "there": ["their", "theyre"],
    "our": ["are"],
    "to": ["too"],
}

# ---------- small helpers for building text + spans ----------

def add_text(text: str, chunk: str) -> str:
    """Append a non-entity chunk with a space (if needed)."""
    chunk = chunk.strip()
    if not chunk:
        return text
    if text:
        return text + " " + chunk
    else:
        return chunk


def add_entity(text: str, spans: List[Dict], chunk: str, label: str) -> str:
    """Append an entity chunk, track its start/end indices (clean text)."""
    chunk = chunk.strip()
    if text:
        start = len(text) + 1  # space
        text = text + " " + chunk
    else:
        start = 0
        text = chunk
    end = start + len(chunk)
    spans.append({"start": start, "end": end, "label": label})
    return text

# ---------- noise helpers (word-level) ----------

def apply_asr_noise(text, prob=0.15):
    words = text.split()
    new_words = []
    for w in words:
        lw = w.lower()
        if lw in ASR_NOISE and random.random() < prob:
            new_words.append(random.choice(ASR_NOISE[lw]))
        else:
            new_words.append(w)
    return " ".join(new_words)


def insert_fillers(text, prob=0.08):
    words = text.split()
    new = []
    for w in words:
        new.append(w)
        if random.random() < prob:
            new.append(random.choice(FILLERS))
    return " ".join(new)


def drop_words(text, prob=0.05):
    words = text.split()
    return " ".join([w for w in words if random.random() > prob])


def random_typo(word):
    if len(word) <= 3:
        return word
    i = random.randint(0, len(word) - 1)
    c = random.choice(string.ascii_lowercase)
    return word[:i] + c + word[i+1:]


def apply_typos(text, prob=0.04):
    return " ".join(
        [random_typo(w) if random.random() < prob else w for w in text.split()]
    )


def spacing_noise(text, prob=0.03):
    """Randomly merge adjacent words."""
    words = text.split()
    new = []
    i = 0
    while i < len(words):
        if i < len(words) - 1 and random.random() < prob:
            new.append(words[i] + words[i + 1])
            i += 2
        else:
            new.append(words[i])
            i += 1
    return " ".join(new)


def apply_homophones(text, prob=0.08):
    words = text.split()
    new = []
    for w in words:
        lw = w.lower()
        if lw in HOMOPHONES and random.random() < prob:
            new.append(random.choice(HOMOPHONES[lw]))
        else:
            new.append(w)
    return " ".join(new)


def shuffle_segments(text, prob=0.10):
    """Shuffle clauses around 'and' occasionally."""
    if random.random() > prob:
        return text
    segments = text.split(" and ")
    if len(segments) <= 1:
        return text
    random.shuffle(segments)
    return " and ".join(segments)


def add_noise(text: str) -> str:
    text = apply_asr_noise(text, prob=0.12)
    text = apply_homophones(text, prob=0.06)
    text = insert_fillers(text, prob=0.05)
    text = drop_words(text, prob=0.02)
    text = apply_typos(text, prob=0.02)
    text = spacing_noise(text, prob=0.01)
    text = shuffle_segments(text, prob=0.05)
    return text


def inject_noise_preserving_entities(text: str, spans: List[Dict]):
    """
    Add noise only in non-entity regions, and recompute spans for the noisy text.
    """
    if not spans:
        noisy_all = add_noise(text)
        return noisy_all, spans

    spans_sorted = sorted(spans, key=lambda s: s["start"])
    pieces = []
    new_spans = []
    prev_end = 0
    current_out_len = 0

    for ent in spans_sorted:
        start, end, label = ent["start"], ent["end"], ent["label"]

        # pre-entity segment (may be empty)
        pre_seg = text[prev_end:start]
        if pre_seg:
            noisy_pre = add_noise(pre_seg)
            pieces.append(noisy_pre)
            current_out_len += len(noisy_pre)

        # entity segment - keep clean
        ent_text = text[start:end]
        new_start = current_out_len
        new_end = new_start + len(ent_text)
        pieces.append(ent_text)
        new_spans.append({"start": new_start, "end": new_end, "label": label})
        current_out_len = new_end
        prev_end = end

    # tail after last entity
    post_seg = text[prev_end:]
    if post_seg:
        noisy_post = add_noise(post_seg)
        pieces.append(noisy_post)
        current_out_len += len(noisy_post)

    noisy_text = "".join(pieces)
    return noisy_text, new_spans

# ---------- entity generators ----------

def make_name():
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    return f"{first} {last}"


def make_email_phrase(name: str) -> str:
    """
    Generate either STT-style or normal email.
    STT: 'ramesh dot sharma at gmail dot com'
    Normal: 'ramesh.sharma@gmail.com' or 'ramesh123@gmail.com'
    """
    first, last = name.split()
    provider = random.choice(EMAIL_PROVIDERS)
    style = random.random()

    # 40%: STT-style (dot/at) - matches assignment examples
    if style < 0.4:
        return f"{first} dot {last} at {provider} dot com"

    # 40%: standard first.last@provider.com or first_last@provider.com
    elif style < 0.8:
        sep = random.choice([".", "_", ""])
        return f"{first}{sep}{last}@{provider}.com"

    # 20%: shorter / numeric variant: first123@provider.com
    else:
        num = random.randint(1, 9999)
        return f"{first}{num}@{provider}.com"



def make_credit_card():
    """16-digit card, grouped into 4 blocks: '4242 4242 4242 4242'"""
    digits = "".join(random.choice("0123456789") for _ in range(16))
    blocks = [digits[i:i + 4] for i in range(0, 16, 4)]
    return " ".join(blocks)


def make_phone_number(spell_prob: float = 0.4):
    """
    Indian-style 10-digit phone, in multiple formats:
    - spelled out: 'nine eight seven six...'
    - plain: '9876543210'
    - grouped: '98765 43210'
    - with country code: '+91 98765 43210'
    - US-style: '(987) 654-3210'
    """
    digits = random.choice("6789") + "".join(
        random.choice("0123456789") for _ in range(9)
    )

    r = random.random()

    # Spelled-out (STT-style)
    if r < spell_prob:
        words = []
        for d in digits:
            if d == "0" and random.random() < 0.3:
                words.append("oh")
            else:
                words.append(NUM_WORDS[d])
        return " ".join(words)

    # Plain digits
    if r < spell_prob + 0.25:
        return digits

    # Grouped 5 + 5
    if r < spell_prob + 0.5:
        return digits[:5] + " " + digits[5:]

    # +91 country code style
    if r < spell_prob + 0.75:
        return "+91 " + digits

    # (987) 654-3210 style
    return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"

def pattern_formal_contact(uid: str) -> Dict:
    """
    More formal / email-style text:
    'for any queries please contact ramesh sharma at ramesh.sharma@gmail.com
     or call +91 98765 43210 in mumbai'
    """
    text = ""
    spans = []

    name = make_name()
    email = make_email_phrase(name)
    phone = make_phone_number(spell_prob=0.2)  # slightly less STT-style
    city = make_city()

    text = add_text(text, "for any queries please contact")
    text = add_entity(text, spans, name, "PERSON_NAME")

    text = add_text(text, "at")
    text = add_entity(text, spans, email, "EMAIL")

    text = add_text(text, "or call")
    text = add_entity(text, spans, phone, "PHONE")

    text = add_text(text, "in")
    text = add_entity(text, spans, city, "CITY")

    return {"id": uid, "text": text, "entities": spans}


def make_date():
    """
    Either numeric: '01 02 2024'
    Or textual: '15 august 2025'
    """
    if random.random() < 0.5:
        d = random.randint(1, 28)
        m = random.randint(1, 12)
        y = random.randint(2023, 2026)
        return f"{d:02d} {m:02d} {y}"
    else:
        d = random.randint(1, 28)
        month_name = random.choice(MONTHS)
        y = random.randint(2023, 2026)
        return f"{d} {month_name} {y}"


def make_city():
    return random.choice(CITIES)


def make_location(city: str):
    """Make a longer location phrase including city: 'andheri east mumbai'"""
    loc = random.choice(LOCATIONS)
    return f"{loc} {city}"

# ---------- utterance templates (CLEAN) ----------

def pattern_card_email_phone(uid: str) -> Dict:
    text = ""
    spans = []

    name = make_name()
    email_phrase = make_email_phrase(name)
    card = make_credit_card()
    phone = make_phone_number()

    text = add_text(text, "my name is")
    text = add_entity(text, spans, name, "PERSON_NAME")

    text = add_text(text, "my email is")
    text = add_entity(text, spans, email_phrase, "EMAIL")

    text = add_text(text, "and my card number is")
    text = add_entity(text, spans, card, "CREDIT_CARD")

    text = add_text(text, "and phone number is")
    text = add_entity(text, spans, phone, "PHONE")

    return {"id": uid, "text": text, "entities": spans}


def pattern_phone_city_date(uid: str) -> Dict:
    text = ""
    spans = []

    phone = make_phone_number()
    city = make_city()
    date = make_date()

    text = add_text(text, "my number is")
    text = add_entity(text, spans, phone, "PHONE")

    text = add_text(text, "i am calling from")
    text = add_entity(text, spans, city, "CITY")

    text = add_text(text, "and i will travel on")
    text = add_entity(text, spans, date, "DATE")

    return {"id": uid, "text": text, "entities": spans}


def pattern_email_only(uid: str) -> Dict:
    text = ""
    spans = []

    name = make_name()
    email_phrase = make_email_phrase(name)

    text = add_text(text, "reach me at")
    text = add_entity(text, spans, email_phrase, "EMAIL")

    return {"id": uid, "text": text, "entities": spans}


def pattern_city_location_date(uid: str) -> Dict:
    text = ""
    spans = []

    city = make_city()
    loc = make_location(city)
    date = make_date()

    text = add_text(text, "i will be in")
    text = add_entity(text, spans, city, "CITY")

    text = add_text(text, "at")
    text = add_entity(text, spans, loc, "LOCATION")

    text = add_text(text, "on")
    text = add_entity(text, spans, date, "DATE")

    return {"id": uid, "text": text, "entities": spans}


def pattern_mixed(uid: str) -> Dict:
    text = ""
    spans = []

    name = make_name()
    email_phrase = make_email_phrase(name)
    card = make_credit_card()
    city = make_city()
    date = make_date()

    text = add_text(text, "email id is")
    if random.random() < 0.5:
        text = add_entity(text, spans, name, "PERSON_NAME")
        text = add_text(text, "and email is")
        text = add_entity(text, spans, email_phrase, "EMAIL")
    else:
        text = add_entity(text, spans, email_phrase, "EMAIL")

    text = add_text(text, "and card number is")
    text = add_entity(text, spans, card, "CREDIT_CARD")

    text = add_text(text, "i will be in")
    text = add_entity(text, spans, city, "CITY")

    text = add_text(text, "on")
    text = add_entity(text, spans, date, "DATE")

    return {"id": uid, "text": text, "entities": spans}
def pattern_email_city(uid: str) -> Dict:
    text = ""
    spans = []

    name = make_name()
    email = make_email_phrase(name)
    city = make_city()
    loc = make_location(city)

    text = add_text(text, "you can email")
    text = add_entity(text, spans, email, "EMAIL")

    text = add_text(text, "i am usually in")
    text = add_entity(text, spans, city, "CITY")

    text = add_text(text, "around")
    text = add_entity(text, spans, loc, "LOCATION")

    return {"id": uid, "text": text, "entities": spans}
def pattern_email_city_location(uid: str) -> Dict:
    text = ""
    spans = []

    name = make_name()
    email = make_email_phrase(name)
    city = make_city()
    loc = make_location(city)

    text = add_text(text, "for any queries you can email")
    text = add_entity(text, spans, email, "EMAIL")

    text = add_text(text, "i am usually in")
    text = add_entity(text, spans, city, "CITY")

    text = add_text(text, "near")
    text = add_entity(text, spans, loc, "LOCATION")

    return {"id": uid, "text": text, "entities": spans}


TEMPLATES = [
    pattern_card_email_phone,
    pattern_phone_city_date,
    pattern_email_only,
    pattern_city_location_date,
    pattern_mixed,    
    pattern_formal_contact, 
    pattern_email_city,  
    pattern_email_city_location,
]

# ---------- core generation helpers ----------

def make_example(idx: int) -> Dict:
    """
    Generate one example:
    1. Build clean text + spans via template
    2. Inject ASR-like noise in non-entity parts
    3. Recompute spans for noisy text
    """
    uid = f"utt_{idx:04d}"
    template = random.choice(TEMPLATES)
    base = template(uid)  # clean
    noisy_text, new_spans = inject_noise_preserving_entities(
        base["text"], base["entities"]
    )
    return {"id": uid, "text": noisy_text, "entities": new_spans}


def write_jsonl(path: str, examples: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def generate_train(n_train: int, train_path: str, start_idx: int = 1) -> int:
    """Generate synthetic train examples and write to file. Returns next idx."""
    train_examples = [make_example(start_idx + i) for i in range(n_train)]
    write_jsonl(train_path, train_examples)
    print(f"Wrote {n_train} train examples to {train_path}")
    return start_idx + n_train


def generate_dev(n_dev: int, dev_path: str, start_idx: int) -> int:
    """Generate synthetic dev examples and write to file. Returns next idx."""
    dev_examples = [make_example(start_idx + i) for i in range(n_dev)]
    write_jsonl(dev_path, dev_examples)
    print(f"Wrote {n_dev} dev examples to {dev_path}")
    return start_idx + n_dev

# ---------- CLI entrypoint ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-train", action="store_true",
                        help="Generate synthetic train.jsonl")
    parser.add_argument("--gen-dev", action="store_true",
                        help="Generate synthetic dev.jsonl")
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-dev", type=int, default=200)
    parser.add_argument("--train-path", type=str, default="data/train.jsonl")
    parser.add_argument("--dev-path", type=str, default="data/dev.jsonl")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Default: if nothing specified, generate TRAIN only
    if not args.gen_train and not args.gen_dev:
        args.gen_train = True

    random.seed(args.seed)
    idx = 1

    if args.gen_train:
        idx = generate_train(args.n_train, args.train_path, start_idx=idx)

    if args.gen_dev:
        idx = generate_dev(args.n_dev, args.dev_path, start_idx=idx)


if __name__ == "__main__":
    main()
