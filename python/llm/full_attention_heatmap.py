import numpy as np
import matplotlib.pyplot as plt


def softmax(x: np.ndarray) -> np.ndarray:
    """Klasyczny softmax po wektorze x (z poprawką numeryczną)."""
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def build_attention_scores(tokens):
    """
    Buduje syntetyczną macierz score'ów dla całej sekwencji.
    Założenia:
      - preferencja lokalna (silniejsza uwaga w okolicach przekątnej),
      - token 'it' patrzy głównie na 'animal' oraz trochę na 'tired'.
    """
    n = len(tokens)
    scores = np.zeros((n, n), dtype=float)

    # 1) Lokalna struktura: im bliżej po przekątnej, tym większy score
    for i in range(n):
        for j in range(n):
            # Ujemna odległość w indeksach -> preferencja dla bliskich pozycji
            scores[i, j] = -abs(i - j)

    # 2) Wzmocnienie wybranych relacji semantycznych
    #    'it' -> 'animal' i w pewnym stopniu 'tired'
    idx_it = tokens.index("it")
    idx_animal = tokens.index("animal")
    idx_tired = tokens.index("tired")

    scores[idx_it, idx_animal] += 3.0  # bardzo mocne powiązanie
    scores[idx_it, idx_tired] += 1.5   # też istotne, ale słabsze

    return scores


def build_attention_matrix(tokens):
    """
    Liczy pełną macierz uwagi α_ij z macierzy score'ów
    poprzez softmax po każdym wierszu.
    """
    scores = build_attention_scores(tokens)
    n = scores.shape[0]
    attn = np.zeros_like(scores)

    for i in range(n):
        attn[i, :] = softmax(scores[i, :])

    return attn


def plot_attention_heatmap(tokens, attn, output_path="attention_heatmap_full.png"):
    """
    Rysuje heatmapę uwagi token->token.
    Wiersze: "kto patrzy"
    Kolumny: "na kogo patrzy"
    """
    n = len(tokens)

    plt.figure(figsize=(7, 6))
    im = plt.imshow(attn, aspect="auto")
    plt.colorbar(im, label="Waga uwagi αᵢⱼ")

    plt.xticks(range(n), tokens, rotation=90)
    plt.yticks(range(n), tokens)

    plt.xlabel("Na który token patrzymy (j)")
    plt.ylabel("Token patrzący (i)")
    plt.title("Przykładowa mapa uwagi dla całej sekwencji")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[OK] Zapisano pełną heatmapę uwagi do: {output_path}")


if __name__ == "__main__":
    # Nasze zdanie:
    tokens = [
        "The", "animal", "did", "not", "cross", "the",
        "street", "because", "it", "was", "tired", "."
    ]

    attn = build_attention_matrix(tokens)
    plot_attention_heatmap(tokens, attn, "attention_heatmap_full.png")
