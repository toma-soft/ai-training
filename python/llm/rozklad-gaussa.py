import numpy as np
import matplotlib.pyplot as plt

# Zakres osi X
x = np.linspace(-5, 5, 500)

# Gęstość prawdopodobieństwa standardowego rozkładu normalnego N(0,1)
pdf = 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2)

plt.figure(figsize=(6, 4))
plt.plot(x, pdf, label="N(0, 1)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.xlabel("x")
plt.ylabel("pdf(x)")
plt.title("Standardowy rozkład normalny Gaussa")
plt.legend()
plt.tight_layout()
plt.show()
