import torch
from model import GPTConfig, MiniGPT
from model.tokenizer_char import CharTokenizer


text = "hello world\n"
tok = CharTokenizer(text)

cfg = GPTConfig(
    vocab_size=tok.vocab_size,
    n_layer=2,
    n_head=2,
    d_model=64,
    d_ff=256,
    block_size=32,
    dropout=0.0,
)

model = MiniGPT(cfg)

# tekst do testu
input_txt = "hello w"
encoded = tok.encode(input_txt)
x = torch.tensor([encoded], dtype=torch.long)  # (B=1, T)

# liczmy stratę
loss = model.compute_loss(x)
logits = model(x)
print("Loss:", loss.item())
print("logits max:", logits.abs().max().item())
print("logits mean:", logits.mean().item())
