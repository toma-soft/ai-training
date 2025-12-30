import numpy as np
import matplotlib.pyplot as plt

def gaussian_bump(t: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - center) / width) ** 2)

if __name__ == "__main__":
    # Sygnał wejściowy (syntetyczny): gładki przebieg z dwoma strukturami
    n = 600
    t = np.linspace(0.0, 1.0, n)

    x = (
        0.9 * np.sin(2.0 * np.pi * 2.0 * t)
        + 0.35 * np.sin(2.0 * np.pi * 7.0 * t + 0.4)
    )

    # "Korekta" F(x): lokalna, umiarkowana modyfikacja (dwa garby + delikatne spłaszczenie)
    f = (
        0.18 * gaussian_bump(t, center=0.30, width=0.05)
        - 0.14 * gaussian_bump(t, center=0.72, width=0.06)
    )

    # Dodatkowo dodajemy korektę zależną od x (żeby nie wyglądało jak czysty offset)
    f += 0.08 * np.tanh(1.2 * x)

    # Warianty wyjścia
    y_plain = f              # y = F(x)
    y_res = x + f            # y = x + F(x)

    # Wykres
    plt.figure()
    plt.plot(t, x, label="wejście x")
    plt.plot(t, y_plain, label="korekta F(x)")
    plt.plot(t, y_res, label="wyjście x + F(x)")

    plt.xlabel("t (oś umowna)")
    plt.ylabel("wartość sygnału (jednostki umowne)")
    plt.title("Porównanie: transformacja kaskadowa i transformacja resztkowa (symulacja)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Jeśli chcesz zapisywać do pliku, odkomentuj:
    # plt.savefig("wykres_2_residual_mechanizm.png", dpi=200)

    plt.show()
