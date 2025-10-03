# 🧠 AI Training

A personal playground for experimenting with Python and AI-related ideas.  
This repository is **not production code** — it’s a lab for testing, learning, and documenting progress.

---

## 📚 Learning Resources

The examples in this repo are based on two books I’m currently studying:

1. *Prosto o AI. Jak działa i myśli sztuczna inteligencja* — Robert Trypuz  
2. *Stwórz własne AI. Jak od podstaw zbudować duży model językowy* — Sebastian Raschka

Reading these is recommended to fully understand the scripts and experiments here.

---

## 🧩 TokenizerV2 (rule-based, with `<|unk|>` & `<|endoftext|>`)

**File:** `python/tokenizer/tokenizerV2.py`

### What it does
- Splits text with a regex that keeps punctuation as separate tokens.
- Maps tokens to integer IDs using a vocabulary built from a sample corpus file.
- Replaces out-of-vocabulary tokens with the special token `<|unk|>`.
- Decodes IDs back to text and fixes spaces before punctuation.

### Special tokens
- `<|unk|>` — used when a token isn’t present in the vocab.  
- `<|endoftext|>` — a delimiter used when concatenating multiple segments.  

These tokens are appended to the vocab that’s built from `the-verdict.txt`.

### How the vocab is prepared
`prepare_vocab()` downloads `the-verdict.txt` from a sample URL and saves it locally.  
`build_vocab_from_file()` then:
- tokenizes the file,
- deduplicates tokens in first-seen order,
- appends `"<|endoftext|>"` and `"<|unk|>"`,
- and builds `{token: id}`.

### Run it (CLI)

```bash
cd python/tokenizer

python3 tokenizerV2.py "This is a sample sentence, with punctuation!"

# Multiple words without quotes also work:
python3 tokenizerV2.py This is a sample sentence with punctuation!