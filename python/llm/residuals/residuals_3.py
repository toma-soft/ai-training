import numpy as np
import matplotlib.pyplot as plt

def simulate_gradient_magnitude(
    depth=120,
    trials=5000,
    mean_j=0.0,
    std_j=0.35,
    seed=7,
):
    """
    Symuluje zachowanie normy gradientu jako iloczynu lokalnych pochodnych.
    - Plain:     g_k = g_{k-1} * |J_k|
    - Residual:  g_k = g_{k-1} * |1 + J_k|
    gdzie J_k ~ N(mean_j, std_j).
    To jest model ilustracyjny, bez wchodzenia w pełny backprop.
    """
    rng = np.random.default_rng(seed)

    g_plain = np.zeros(depth + 1, dtype=float)
    g_res = np.zeros(depth + 1, dtype=float)

    for _ in range(trials):
        gp = 1.0
        gr = 1.0

        g_plain[0] += gp
        g_res[0] += gr

        for k in range(1, depth + 1):
            j = rng.normal(loc=mean_j, scale=std_j)

            gp *= abs(j)
            gr *= abs(1.0 + j)

            g_plain[k] += gp
            g_res[k] += gr

    g_plain /= trials
    g_res /= trials

    return g_plain, g_res

if __name__ == "__main__":
    depth = 120
    trials = 5000

    plain, res = simulate_gradient_magnitude(
        depth=depth,
        trials=trials,
        mean_j=0.0,
        std_j=0.25,
        seed=11,
    )

    x = np.arange(depth + 1)

    plt.figure()
    plt.plot(x, plain, label="bez połączeń resztkowych")
    plt.plot(x, res, label="z połączeniami resztkowymi")

    plt.xlabel("Głębokość (liczba warstw/bloków)")
    plt.ylabel("Uśredniona wartość |∂y/∂x| (jednostki umowne)")
    plt.title("Wpływ połączeń resztkowych na propagację gradientu (symulacja)")
    plt.yscale("log")  # log pomaga, bo w plain często jest bardzo szybki zanik
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Jeśli chcesz zapisać:
    # plt.savefig("wykres_3_gradient_residuals.png", dpi=200)

    plt.show()
