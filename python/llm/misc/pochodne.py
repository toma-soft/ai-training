import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)

    y = np.sin(x)
    dy = np.cos(x)        # pierwsza pochodna
    ddy = -np.sin(x)      # druga pochodna

    plt.figure()
    plt.plot(x, y, label="sin(x)")
    plt.plot(x, dy, label="sin'(x) = cos(x)")
    plt.plot(x, ddy, label="sin''(x) = -sin(x)")

    plt.xlabel("x")
    plt.ylabel("wartość (jednostki umowne)")
    plt.title("Funkcja sinus i jej pochodne")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()
