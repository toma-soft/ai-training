import numpy as np
import matplotlib.pyplot as plt


def softmax(x: np.ndarray) -> np.ndarray:
    """Klasyczny softmax z poprawką numeryczną."""
    e = np.exp(x - np.max(x))
    return e / e.sum()


def plot_softmax_scaling(output_path: str = "softmax_scaling.png") -> None:
    """
    Rysuje wykres pokazujący wpływ skalowania 1/sqrt(d_k) na softmax.
    """
    # Przykładowe score'y (np. z QK^T)
    scores = np.linspace(-5, 5, 9)

    # Załóżmy d_k = 4 (tylko poglądowo)
    d_k = 4
    scaled_scores = scores / np.sqrt(d_k)

    soft = softmax(scores)
    soft_scaled = softmax(scaled_scores)

    plt.figure()
    plt.plot(scores, soft, marker="o", label="softmax(scores)")
    plt.plot(scores, soft_scaled, marker="x", linestyle="--",
             label="softmax(scores / √d_k)")
    plt.xlabel("score")
    plt.ylabel("softmax(score)")
    plt.title("Wpływ skalowania 1/√d_k na softmax")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[OK] Zapisano wykres softmaxu do: {output_path}")


def plot_attention_heatmap(output_path: str = "attention_heatmap_it_animal.png") -> None:
    """
    Rysuje przykładową mapę uwagi dla zdania:
    'The animal did not cross the street because it was tired.'
    Wiersz odpowiadający 'it' ma najwyższą wagę na 'animal'.
    """
    tokens = [
        "The", "animal", "did", "not", "cross", "the",
        "street", "because", "it", "was", "tired", "."
    ]
    n = len(tokens)

    # Startujemy od prawie jednolitej uwagi
    attn = np.full((n, n), 0.02, dtype=float)

    # Znajdujemy indeks tokenu 'it'
    row_it = tokens.index("it")

    # Ręcznie ustawiony rozkład uwagi dla 'it'
    # (większa masa na 'animal' i trochę na 'tired')
    weights = np.array([
        0.02,  # The
        0.45,  # animal  <-- największa waga
        0.03,  # did
        0.03,  # not
        0.02,  # cross
        0.02,  # the
        0.05,  # street
        0.05,  # because
        0.10,  # it
        0.06,  # was
        0.15,  # tired
        0.02,  # .
    ], dtype=float)

    # Normalizacja na wszelki wypadek
    weights /= weights.sum()

    attn[row_it, :] = weights

    plt.figure(figsize=(6, 5))
    plt.imshow(attn, aspect="auto")
    plt.colorbar(label="Waga uwagi αᵢⱼ")
    plt.xticks(range(n), tokens, rotation=90)
    plt.yticks(range(n), tokens)
    plt.title("Przykładowa mapa uwagi\n(wiersz 'it' skupiony na 'animal')")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[OK] Zapisano heatmapę uwagi do: {output_path}")


if __name__ == "__main__":
    plot_softmax_scaling("softmax_scaling.png")
    plot_attention_heatmap("attention_heatmap_it_animal.png")
