import numpy as np
import matplotlib.pyplot as plt

def simulate_signal_norms(
    depth=100,
    dim=256,
    trials=200,
    nonlinearity="tanh",
    residual_scale=0.002,
    weight_scale=1.0,
    seed=42,
):
    rng = np.random.default_rng(seed)

    def phi(x):
        if nonlinearity == "tanh":
            return np.tanh(x)
        if nonlinearity == "relu":
            return np.maximum(0.0, x)
        raise ValueError("Unsupported nonlinearity")

    plain_norms = np.zeros(depth + 1, dtype=float)
    res_norms = np.zeros(depth + 1, dtype=float)

    for _ in range(trials):
        x0 = rng.normal(size=(dim,))
        x0 = x0 / (np.linalg.norm(x0) + 1e-12)

        x_plain = x0.copy()
        x_res = x0.copy()
        x_res = x_res / (np.linalg.norm(x_res) + 1e-12)

        plain_norms[0] += np.linalg.norm(x_plain)
        res_norms[0] += np.linalg.norm(x_res)

        for k in range(1, depth + 1):
            W = rng.normal(size=(dim, dim)) / np.sqrt(dim)
            W = W * weight_scale

            x_plain = phi(W @ x_plain)
            x_res = x_res + residual_scale * phi(W @ x_res)

            plain_norms[k] += np.linalg.norm(x_plain)
            res_norms[k] += np.linalg.norm(x_res)

    plain_norms /= trials
    res_norms /= trials

    return plain_norms, res_norms

if __name__ == "__main__":
    depth = 120
    dim = 256
    trials = 200

    plain, res = simulate_signal_norms(
        depth=depth,
        dim=dim,
        trials=trials,
        nonlinearity="tanh",
        residual_scale=0.1,
        weight_scale=1.0,
        seed=7,
    )

    x = np.arange(depth + 1)

    plt.figure()
    plt.plot(x, plain, label="bez połączeń resztkowych")
    plt.plot(x, res, label="z połączeniami resztkowymi")
    plt.xlabel("Głębokość (liczba warstw/bloków)")
    plt.ylabel("Średnia norma sygnału")
    plt.title("Wpływ głębokości na propagację sygnału (symulacja)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
