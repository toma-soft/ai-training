import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi

# Przybliżenie GeLU takie jak w GPT
def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(
        sqrt(2.0 / pi) * (x + 0.044715 * np.power(x, 3))
    ))

def relu(x):
    return np.maximum(0, x)

x = np.linspace(-5, 5, 500)

plt.figure(figsize=(6, 4))
plt.plot(x, relu(x), label="ReLU")
plt.grid(True, linestyle="--", alpha=0.5)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Funkcja ReLU")
plt.legend()
plt.tight_layout()
plt.show()
