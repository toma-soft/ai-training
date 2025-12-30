import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)


def layer_norm(x, eps=1e-5):
    """
    LayerNorm po ostatnim wymiarze (cechy tokenu).
    Zwraca x_hat, mean, var.
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

# kilka outlierów (dydaktycznie celowo)
x[:4] += np.array([25, -22, 18, -16])

x = x.reshape(1, d)


# =========================================================
# 2) LayerNorm: normalizacja do x_hat
# =========================================================

x_hat, mu, var = layer_norm(x)

print("Przed LayerNorm:  mean=%.4f  std=%.4f" % (x.mean(), x.std()))
print("Po LayerNorm:     mean=%.4f  std=%.4f" % (x_hat.mean(), x_hat.std()))


# =========================================================
# 3) Konfiguracje γ i β
# =========================================================

configs = [
    (1.0, 0.0, "γ=1.0, β=0.0 (referencja)"),
    (0.5, 0.0, "γ=0.5, β=0.0 (kompresja skali)"),
    (2.0, 0.0, "γ=2.0, β=0.0 (wzmocnienie skali)"),
    (1.0, 1.0, "γ=1.0, β=1.0 (przesunięcie średniej)"),
]


# =========================================================
# 4) Wykres 1: histogram „step” (gęstość) + log Y
#    Zakres X szeroki [-6, 6] (ogony będą blisko zera – i o to chodzi)
# =========================================================

hist_x_min, hist_x_max = -6.0, 6.0
bins_hist = np.linspace(hist_x_min, hist_x_max, 100)

plt.figure(figsize=(11, 6))

for gamma, beta, label in configs:
    y = x_hat * gamma + beta
    plt.hist(
        y.flatten(),
        bins=bins_hist,
        density=True,
        histtype="step",
        linewidth=2,
        label=label
    )

plt.axvline(0, linestyle="--", linewidth=1)
plt.xlim(hist_x_min, hist_x_max)
plt.yscale("log")
plt.xlabel("wartość cechy")
plt.ylabel("gęstość (skala log)")
plt.title("Wpływ parametrów γ i β na rozkład aktywacji po LayerNorm")
plt.legend()
plt.tight_layout()
plt.show()


# =========================================================
# 5) Wykres 2: CDF (ten „kozacki”)
#    Zakres węższy dla czytelności kształtu (tu to naprawdę działa)
# =========================================================

cdf_x_min, cdf_x_max = -3.5, 3.5

plt.figure(figsize=(11, 6))

for gamma, beta, label in configs:
    y = x_hat * gamma + beta
    values = np.sort(y.flatten())
    cdf = np.arange(1, len(values) + 1) / len(values)
    plt.plot(values, cdf, label=label)

plt.axvline(0, linestyle="--", linewidth=1)
plt.xlim(cdf_x_min, cdf_x_max)
plt.xlabel("wartość cechy")
plt.ylabel("dystrybuanta (CDF)")
plt.title("Wpływ parametrów γ i β: dystrybuanta aktywacji po LayerNorm")
plt.legend()
plt.tight_layout()
plt.show()
