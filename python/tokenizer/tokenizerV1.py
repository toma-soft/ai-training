import re
import sys
from collections import OrderedDict


class TokenizerV1:
    def __init__(self, vocab):
        self.str_to_init = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
        
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_init[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\]])', r'\1', text)
        return text


def build_vocab_from_text(text: str) -> dict[str, int]:
    """
    Build a {token: id} vocab from the given text using the *same* split
    logic as the tokenizer, preserving first-seen order.
    """
    tokens = re.split(r'([,.?_!"()\']|--|\s)', text)
    tokens = [t.strip() for t in tokens if t.strip()]
    # preserve order of first appearance
    unique_tokens = list(OrderedDict.fromkeys(tokens))
    return {tok: idx for idx, tok in enumerate(unique_tokens)}


def run_tokenizer_v1(text):

    # Auto-build a vocab from the provided text,
    # so every token is guaranteed to be known.
    vocab = build_vocab_from_text(text)

    tokenizer = TokenizerV1(vocab)

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    print("=== Input Text ===")
    print(text)
    print("\n=== Vocab (string -> id) ===")
    print(vocab)
    print("\n=== Encoded IDs ===")
    print(encoded)
    print("\n=== Decoded Text ===")
    print(decoded)
    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tokenizer.py \"Your text here\"")
        sys.exit(1)
    # Allow passing multiple CLI args without quotes
    text = " ".join(sys.argv[1:])
    run_tokenizer_v1(text)
