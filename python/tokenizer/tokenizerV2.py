import re
import sys
import urllib.request
from collections import OrderedDict


class TokenizerV2:
    def __init__(self, vocab):
        self.str_to_init = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [
            item if item in self.str_to_init
            else "<|unk|>" for item in preprocessed
        ]
        ids = [self.str_to_init[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\]])', r'\1', text)
        return text


def prepare_vocab():
    url = "https://cnb.cool/8888/github.com/MLNLP-World/LLMs-from-scratch-CN/-/git/raw/8dc2f15e4cc3b68a30e74eedcf48387e32689e30/the-verdict.txt"  # noqa: E501
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)


def print_vocab():
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
        print("Total length: ", len(raw_text))
        print(raw_text[:99])


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


def build_vocab_from_file() -> dict[str, int]:
    raw_text = ""
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    tokens = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
    tokens = [t.strip() for t in tokens if t.strip()]
    # preserve order of first appearance
    unique_tokens = list(OrderedDict.fromkeys(tokens))
    unique_tokens.extend(["<|endoftext|>", "<|unk|>"])
    return {tok: idx for idx, tok in enumerate(unique_tokens)}


def run_tokenizer_v2(text):

    # Auto-build a vocab from the provided text,
    # so every token is guaranteed to be known.
    vocab = build_vocab_from_file()

    tokenizer = TokenizerV2(vocab)

    encoded = tokenizer.encode(text)
    print(encoded)
    decoded = tokenizer.decode(encoded)
    print(decoded)
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
    second_fixed_text = "eins zwei drei"
    input_text = " ".join(sys.argv[1:])
    text = " <|endoftext|> ".join([input_text, second_fixed_text])
    prepare_vocab()
    run_tokenizer_v2(text)
