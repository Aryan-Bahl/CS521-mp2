import pandas as pd
import matplotlib.pyplot as plt
import os

print("pwd: ", os.getcwd())
df = pd.read_csv("results/inductor_vs_reference.csv")

# X-axis labels
x_labels = [f"{row['in_shape']}|{row['filter']}" for _, row in df.iterrows()]
x = range(len(df))

plt.figure(figsize=(12,6))
# 3 bars side by side: Inductor compile, Inductor kernel, Reference wall
plt.bar([i-0.2 for i in x], df['inductor_wall_ms'], width=0.2, label="Inductor Wall", alpha=0.8)
plt.bar([i for i in x], df['inductor_kernel_ms'], width=0.2, label="Inductor Kernel", alpha=0.8)
plt.bar([i+0.2 for i in x], df['reference_wall_ms'], width=0.2, label="Reference Wall", alpha=0.8)

plt.xticks(x, x_labels, rotation=45, ha='right')
plt.ylabel("Time (ms)")
plt.yscale('log')
plt.title("Conv2D Inductor vs Reference")
plt.legend()
plt.tight_layout()
plt.savefig("results/inductor_vs_reference.png")
plt.show()