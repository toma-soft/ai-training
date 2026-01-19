import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Zabawkowe współrzędne embeddingów w 3D
points = {
    "Ala": (1.4, 2.2, 0.9),
    "ma": (2.8, 0.7, 0.4),
    "mysz": (4.6, 2.0, 1.0),
    "myszoskoczek": (5.0, 1.8, 0.85),  # lekko opuszczony
}

origin = (0.0, 0.0, 0.0)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# Kolory: mysz + myszoskoczek wyróżnione
colors = {
    "Ala": "tab:blue",
    "ma": "tab:blue",
    "mysz": "tab:red",
    "myszoskoczek": "tab:red",
}

# Punkty + etykiety
for label, (x, y, z) in points.items():
    ax.scatter(x, y, z, s=80, color=colors[label])
    ax.text(x + 0.1, y + 0.05, z + 0.05, label, fontsize=12)

# Strzałki od (0,0,0) → punkt
for label, (x, y, z) in points.items():
    ax.quiver(
        origin[0], origin[1], origin[2],
        x, y, z,
        arrow_length_ratio=0.08,
        linewidth=2,
        color=colors[label],
        normalize=False
    )

ax.set_title('Przykładowa wizualizacja embeddingów w 3D: "Ala ma myszoskoczka."', fontsize=13)

# Opisy osi
ax.set_xlabel("wymiar X")
ax.set_ylabel("wymiar Y")
ax.set_zlabel("wymiar Z")

# Zakresy osi – 0,0,0 po lewej
ax.set_xlim(0, 6)
ax.set_ylim(0, 3.5)
ax.set_zlim(0, 2)

# Kamera: punkt (0,0,0) po lewej, wektory idą w prawo
ax.view_init(elev=18, azim=-60)

plt.tight_layout()
plt.show()
