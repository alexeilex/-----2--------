import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Чтение данных
df = pd.read_csv("solution_3d.csv")

# Перестройка в матричный вид (x, y уникальны и упорядочены)
x_unique = sorted(df["x"].unique())
y_unique = sorted(df["y"].unique())
X, Y = np.meshgrid(x_unique, y_unique)

# Извлечение решений
num = df.pivot(index="y", columns="x", values="numerical").values
exact = df.pivot(index="y", columns="x", values="exact").values

# Построение
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X, Y, num, cmap='viridis', alpha=0.9)
ax1.set_title("Numerical Solution")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("u")

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(X, Y, exact, cmap='plasma', alpha=0.9)
ax2.set_title("Exact Solution")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("u")

plt.tight_layout()
plt.savefig("solution_3d.png", dpi=150)
plt.show()