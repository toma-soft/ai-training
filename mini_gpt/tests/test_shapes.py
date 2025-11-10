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

x = torch.tensor([tok.encode("hello")], dtype=torch.long)  # (1,5)
logits = model(x)
assert logits.shape == (1, 5, tok.vocab_size)
print("logits OK:", logits.shape)
