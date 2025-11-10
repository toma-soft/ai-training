import torch
from model import GPTConfig, MiniGPT
from model.tokenizer_char import CharTokenizer

txt = "hello world from MiniGPT\n"
tok = CharTokenizer(txt)

cfg = GPTConfig(
    vocab_size=tok.vocab_size, n_layer=2, n_head=2,
    d_model=64, d_ff=256, block_size=32, dropout=0.0
)
model = MiniGPT(cfg)

x = torch.tensor([tok.encode("hello ")], dtype=torch.long)
with torch.no_grad():
    out = model.generate(x, max_new_tokens=20, top_k=10)
print("len:", out.shape[1])
print("sample:", tok.decode(out[0].tolist()))
