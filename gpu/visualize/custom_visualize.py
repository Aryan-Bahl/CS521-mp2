import pandas as pd
import matplotlib.pyplot as plt
import os

print("pwd: ", os.getcwd())
df = pd.read_csv("results/custom_vs_reference.csv")

# X-axis labels
x_labels = [f"{row['in_shape']}|{row['filter']}" for _, row in df.iterrows()]
x = range(len(df))

plt.figure(figsize=(12,6))
# 2 bars side by side: Custom CUDA kernel vs Reference
plt.bar([i-0.2 for i in x], df['custom_wall_ms'], width=0.4, label="Custom CUDA", alpha=0.8)
plt.bar([i+0.2 for i in x], df['reference_wall_ms'], width=0.4, label="Reference", alpha=0.8)

plt.xticks(x, x_labels, rotation=45, ha='right')
plt.ylabel("Time (ms)")
plt.yscale('log')
plt.title("Conv2D Custom CUDA vs Reference")
plt.legend()
plt.tight_layout()
plt.savefig("results/custom_vs_reference.png")
plt.show()
