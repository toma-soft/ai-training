# 🧩 miniGPT — minimalny model językowy w Pythonie

Ten projekt to **edukacyjna implementacja GPT** napisana w czystym Pythonie + PyTorch.  
Celem jest **zrozumienie każdego elementu transformera**, od embeddingów po generację tekstu.

---

## 📁 Struktura projektu

```
mini_gpt/
 ├── __init__.py
 ├── tokenizer_char.py         # prosty tokenizer znakowy
 └── model/
     ├── __init__.py
     ├── config.py             # konfiguracja modelu (rozmiary, warstwy)
     ├── token_embedding.py    # zamiana indeksów tokenów na wektory
     ├── positional_embedding.py  # dodaje informację o pozycji
     ├── multi_head_attention.py  # rdzeń transformera (self-attention)
     ├── feed_forward.py       # sieć MLP po attention
     ├── block.py              # pojedynczy blok transformera
     └── gpt.py                # pełny model MiniGPT z generate() i compute_loss()

tests/
 ├── test_shapes.py            # sprawdza poprawność wymiarów
 ├── run_generate.py           # generuje przykładowy tekst
 └── train_step.py             # prosty trening (opcjonalny)
```

---

## ⚙️ Przepływ danych

```
idx (B,T)
  │
  ▼
TokenEmbedding (B,T,d_model)
  │
  ▼
PositionalEmbedding
  │
  ▼
N × [LayerNorm → MultiHeadAttention → +residual → LayerNorm → FeedForward → +residual]
  │
  ▼
LayerNorm
  │
  ▼
Linear (lm_head)
  │
  ▼
logits (B,T,vocab)
```

---

## 🧠 Diagramy

### a) Attention z maską przyczynową

```
                ┌────────────────────────────┐
Input x ───────►│ Linear(3*d_model) → [Q,K,V]│
                └────────────┬───────────────┘
                             ▼
                    ┌────────────────┐
                    │  podział na    │
                    │  n_head głów   │
                    └────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
         Q₁                 Q₂                 Qₙ
         K₁                 K₂                 Kₙ
         V₁                 V₂                 Vₙ
          │                  │                  │
          ▼                  ▼                  ▼
        QKᵀ/√d ⟶ mask upper-tri  (zero future)
          │
          ▼
      softmax(attention weights)
          │
          ▼
        A·V → głowa wynikowa
          │
          ▼
    concat wszystkich głów
          │
          ▼
   Linear projection → (B,T,d_model)
          │
          ▼
   Dropout + Residual + LayerNorm
```

➡️ Maska trójkątna (`triu`) gwarantuje, że token *t* nie widzi tokenów *t+1…T*.

---

### b) Shift w `compute_loss()`

```
Wejście:  idx = [h, e, l, l, o]
Logits:   model przewiduje prawdopodobieństwo KAŻDEGO tokenu
          ┌───┬───┬───┬───┬───┐
          │h→?│e→?│l→?│l→?│o→?│
          └───┴───┴───┴───┴───┘

Targets:  przesunięcie o 1 w lewo
          [e, l, l, o, <pad>]

Loss liczymy między:
    logits[:, :-1, :]  vs  targets[:, 1:]
czyli model uczy się przewidywać NASTĘPNY znak
```

---

## 🚀 Uruchomienie

```bash
pip install torch
python -m tests.test_shapes
python -m tests.run_generate
```

*(opcjonalnie)* trening:
```bash
python -m tests.train_step
```

---

## 🎯 Cel projektu

- Zrozumieć **wewnętrzną logikę transformera GPT**.  
- Mieć własny, w pełni czytelny kod od embeddingów po sampling.  
- Móc potem samodzielnie rozwijać model (RAG, BPE, funkcje, itp.).

---

## 📚 Autor

Projekt edukacyjny Macieja — nauka LLM „od środka”.
