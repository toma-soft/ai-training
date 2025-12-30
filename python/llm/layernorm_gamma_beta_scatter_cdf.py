import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)


def layer_norm(x, eps=1e-5):
    """
    Layer Normalization po ostatnim wymiarze (cechy tokenu).
    Zwraca: x_hat, mean, var
    """
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_hat = (x - mu) / np.sqrt(var + eps)
    return x_hat, mu, var


# =========================================================
# 1) Dane wejściowe: jeden token o rozjechanych skalach
# =========================================================

d = 256

x = np.concatenate([
    np.random.normal(loc=6.0,  scale=6.0,  size=d // 2),
    np.random.normal(loc=-3.0, scale=0.6, size=d // 2),
], axis=0)

# kilka outlierów (celowo, dydaktycznie)
x[:4] += np.array([25, -22, 18, -16])

x = x.reshape(1, d)


# =========================================================
# 2) LayerNorm
# =========================================================

x_hat, mu, var = layer_norm(x)

print("Przed LayerNorm: mean=%.4f  std=%.4f" % (x.mean(), x.std()))
print("Po LayerNorm:    mean=%.4f  std=%.4f" % (x_hat.mean(), x_hat.std()))


# =========================================================
# 3) Konfiguracje γ i β
# =========================================================

scatter_configs = [
    (1.0, 0.0, "γ=1.0 (referencja)"),
    (0.5, 0.0, "γ=0.5 (kompresja skali)"),
    (2.0, 0.0, "γ=2.0 (wzmocnienie skali)"),
]

cdf_configs = [
    (1.0, 0.0, "γ=1.0, β=0.0 (referencja)"),
    (0.5, 0.0, "γ=0.5, β=0.0 (kompresja skali)"),
    (2.0, 0.0, "γ=2.0, β=0.0 (wzmocnienie skali)"),
    (1.0, 1.0, "γ=1.0, β=1.0 (przesunięcie średniej)"),
]


# =========================================================
# 4) WYKRES 1 — ROZRZUT PUNKTOWY (SCATTER)
# =========================================================

plt.figure(figsize=(11, 6))

idx = np.arange(x_hat.shape[1])

for gamma, beta, label in scatter_configs:
    y = x_hat * gamma + beta
    x_jitter = y.flatten() + np.random.normal(scale=0.1, size=y.size)
    plt.scatter(
        y.flatten(),
        idx,
        s=14,
        alpha=0.6,
        label=label
    )

plt.axvline(0, linestyle="--", linewidth=1)
plt.xlim(-6, 6)
plt.xlabel("wartość cechy")
plt.ylabel("indeks cechy")
plt.title("Rozrzut aktywacji po LayerNorm dla różnych wartości parametru γ")
plt.legend()
plt.tight_layout()
plt.show()


# =========================================================
# 5) WYKRES 2 — CDF (TEN „KOZACKI”)
# =========================================================

plt.figure(figsize=(11, 6))

for gamma, beta, label in cdf_configs:
    y = x_hat * gamma + beta
    values = np.sort(y.flatten())
    cdf = np.arange(1, len(values) + 1) / len(values)
    plt.plot(values, cdf, label=label)

plt.axvline(0, linestyle="--", linewidth=1)
plt.xlim(-3.5, 3.5)
plt.xlabel("wartość cechy")
plt.ylabel("dystrybuanta (CDF)")
plt.title("Wpływ parametrów γ i β na dystrybuantę aktywacji po LayerNorm")
plt.legend()
plt.tight_layout()
plt.show()
