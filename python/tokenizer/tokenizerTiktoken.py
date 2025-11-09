import sys
import tiktoken


def run_tokenizer_tiktoken(text):

    # Auto-build a vocab from the provided text,
    # so every token is guaranteed to be known.
    # vocab = build_vocab_from_file()

    tokenizer = tiktoken.get_encoding("gpt2")

    encoded = tokenizer.encode(text)
    print(encoded)
    decoded = tokenizer.decode(encoded)
    print(decoded)
    print("=== Input Text ===")
    print(text)
    # print("\n=== Vocab (string -> id) ===")
    # print(vocab)
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
    # prepare_vocab()
    run_tokenizer_tiktoken(text)
