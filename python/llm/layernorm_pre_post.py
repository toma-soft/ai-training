import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)

def layer_norm_vec(x, eps=1e-5):
    mu = x.mean()
    var = x.var()
    return (x - mu) / np.sqrt(var + eps)

def run(depth=64, d=256, mode="pre"):
    # losowy wektor wejściowy
    x = np.random.normal(size=(d,))
    # "waga" bloku jako losowa macierz (stała)
    W = np.random.normal(scale=0.05, size=(d, d))

    # forward: składamy bloki
    xs = [x.copy()]
    for _ in range(depth):
        if mode == "pre":
            h = layer_norm_vec(xs[-1])
            f = W @ h
            y = xs[-1] + f
        else:  # post
            f = W @ xs[-1]
            y = layer_norm_vec(xs[-1] + f)
        xs.append(y)

    # backward (bardzo uproszczone): śledzimy jak rośnie/zanika "sygnał gradientu"
    # Tu nie liczymy prawdziwych pochodnych LN (to byłoby długie),
    # tylko pokazujemy efekt wielokrotnego "mieszania" sygnału.
    g = np.ones_like(xs[-1])
    norms = [np.linalg.norm(g)]

    for i in reversed(range(depth)):
        if mode == "pre":
            # residual daje bezpośrednią składową gradientu
            g = g + (W.T @ g) * 0.5
        else:
            # post: gradient jest "przefiltrowany" częściej
            g = (g + (W.T @ g) * 0.5) * 0.9
        norms.append(np.linalg.norm(g))

    norms.reverse()
    return norms

depth = 96
pre = run(depth=depth, mode="pre")
post = run(depth=depth, mode="post")

plt.figure()
plt.plot(range(len(pre)), pre, label="Pre-LN (toy)")
plt.plot(range(len(post)), post, label="Post-LN (toy)")
plt.yscale("log")
plt.title("Wizualizacja: stabilność przepływu gradientu (zabawka)")
plt.xlabel("warstwa")
plt.ylabel("||g|| (skala log)")
plt.legend()
plt.tight_layout()
plt.show()
