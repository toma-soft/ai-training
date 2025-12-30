import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)

def layer_norm(x, eps=1e-5, gamma=None, beta=None):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_hat = (x - mu) / np.sqrt(var + eps)
    if gamma is None:
        gamma = np.ones((1, x.shape[-1]))
    if beta is None:
        beta = np.zeros((1, x.shape[-1]))
    return x_hat * gamma + beta, x_hat, mu, var

# "Token" jako wektor cech: mieszanka skal (symulacja driftu)
d = 256
x = np.concatenate([
    np.random.normal(loc=6.0,  scale=6.0,  size=d//2),   # szeroki komponent
    np.random.normal(loc=-3.0, scale=0.6, size=d//2),    # wąski komponent
], axis=0)

# opcjonalnie: outliery (bardzo dydaktyczne)
x[:4] += np.array([25, -22, 18, -16])

x = x.reshape(1, d)
y, x_hat, mu, var = layer_norm(x)
# przed LayerNorm
plt.figure()
plt.hist(x.flatten(), bins=40)
plt.xlim(-10, 10)
plt.title("Przed LayerNorm: histogram cech tokenu")
plt.xlabel("wartość cechy")
plt.ylabel("liczność")
plt.tight_layout()
plt.show()

# po LayerNorm
plt.figure()
plt.hist(x_hat.flatten(), bins=40)
plt.xlim(-4, 4)
plt.title("Po normalizacji (x̂): średnia≈0, wariancja≈1")
plt.xlabel("wartość cechy")
plt.ylabel("liczność")
plt.tight_layout()
plt.show()

print("Przed: mean=%.4f var=%.4f" % (x.mean(), x.var()))
print("Po (x_hat): mean=%.4f var=%.4f" % (x_hat.mean(), x_hat.var()))
