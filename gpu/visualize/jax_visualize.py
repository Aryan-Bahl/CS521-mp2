import pandas as pd
import matplotlib.pyplot as plt
import os

print("pwd: ", os.getcwd())
df = pd.read_csv("results/jax_vs_torch.csv")

# X-axis labels
x_labels = [f"{row['in_shape']}|{row['filter']}" for _, row in df.iterrows()]
x = range(len(df))

plt.figure(figsize=(12,6))
# 3 bars side by side: JAX compile, JAX kernel, PyTorch wall
plt.bar([i-0.2 for i in x], df['jax_wall_ms'], width=0.2, label="JAX Wall", alpha=0.8)
plt.bar([i for i in x], df['jax_kernel_ms'], width=0.2, label="JAX Kernel", alpha=0.8)
plt.bar([i+0.2 for i in x], df['torch_wall_ms'], width=0.2, label="PyTorch Wall", alpha=0.8)

plt.xticks(x, x_labels, rotation=45, ha='right')
plt.ylabel("Time (ms)")
plt.yscale('log')
plt.title("Conv2D JAX vs PyTorch")
plt.legend()
plt.tight_layout()
plt.savefig("results/jax_vs_torch.png")
plt.show()
