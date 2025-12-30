import torch
from model import GPTConfig, MiniGPT
from model.tokenizer_char import CharTokenizer

txt = "hello world\n"
tok = CharTokenizer(txt)

cfg = GPTConfig(
    vocab_size=tok.vocab_size, n_layer=2, n_head=2,
    d_model=64, d_ff=256, block_size=32, dropout=0.0
)
model = MiniGPT(cfg)
input_txt = "hello w"
encoded = tok.encode(input_txt)
x = torch.tensor([encoded], dtype=torch.long)  # (1,5)
B, T = x.shape
logits = model(x)
assert logits.shape == (B, T, tok.vocab_size)
print("logits OK:", logits.shape)
