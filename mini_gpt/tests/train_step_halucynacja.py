import random
import torch
from model import GPTConfig, MiniGPT
from model.tokenizer_char import CharTokenizer


def set_seed(seed=1337):
    random.seed(seed)
    torch.manual_seed(seed)


def make_batches(ids, block_size, batch_size):
    """
    Z losowych pozycji wycinamy sekwencje długości block_size.
    Zwraca batch (B, T).
    """
    ix = torch.randint(low=0, high=len(ids) - block_size - 1, size=(batch_size,))
    x = torch.stack([torch.tensor(ids[i:i+block_size]) for i in ix])
    return x.long()


def main():
    set_seed()

    # mały korpus – możesz podmienić na dłuższy tekst
    text = (
        "Cat is big\n"
        "Kitty is tiny\n"
        "Dog is big\n"
        "Puppy is a tiny dog\n"
        "Dogs do not like cats\n"
        "Cats do not like dogs\n"
        "Cat Cat Cat - are you drunk buddy ?"
        "Cat Cat Cat Cat - are you drunk buddy ?"
        "Are you drunk buddy ?"
        "You are drunk buddy!"
        "buddy"
        "buddy buddy buddy buddy buddy"
        "buddy"
        "buddy"
        "buddy"
        "buddy"
        "buddy"
    )
    tok = CharTokenizer(text)

    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        n_layer=2, n_head=2,
        d_model=256, d_ff=256,
        block_size=32, dropout=0.0
    )
    model = MiniGPT(cfg)

    # hiperparametry
    lr = 1e-3
    steps = 300
    batch_size = 32

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    ids = tok.encode(text)

    for step in range(1, steps + 1):
        x = make_batches(ids, cfg.block_size, batch_size)  # (B, T)
        loss = model.compute_loss(x)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 100 == 0 or step == 1:
            print(f"step {step:4d} | loss {loss.item():.4f}")

    # szybka próbka po treningu
    model.eval()
    with torch.no_grad():
        # TEST HALUCYNACJI: MiniGPT próbuje odpowiedzieć na pytanie matematyczne
        print("\n=== TEST HALUCYNACJI MATEMATYCZNEJ ===")
        test_prompt = "Cat Cat Cat"
        ids = tok.encode(test_prompt)
        x = torch.tensor([ids], dtype=torch.long)

        out = model.generate(x, max_new_tokens=30, top_k=None)[0].tolist()
        answer = tok.decode(out[len(ids):])  # wszystko po pytaniu
        print("prompt:", repr(test_prompt))
        print("model answer:", repr(answer))


if __name__ == "__main__":
    main()
