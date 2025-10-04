import pandas as pd
import matplotlib.pyplot as plt
import os

print("pwd: ", os.getcwd())
df = pd.read_csv("results/myconv_vs_baseline.csv")

# X-axis labels
x_labels = [f"{row['in_shape']}|{row['filter']}" for _, row in df.iterrows()]
x = range(len(df))

plt.figure(figsize=(10,6))
plt.bar([i+0.2 for i in x], df['manual_ms'],   width=0.4, label="Manual (im2col+GEMM)", alpha=0.8)
plt.bar([i-0.2 for i in x], df['baseline_ms'], width=0.4, label="Baseline (cuDNN)", alpha=0.8)

plt.xticks(x, x_labels, rotation=45, ha='right')
plt.ylabel("Time (ms)")
plt.yscale('log')
plt.title("Conv2D Baseline vs Manual CUDA Implementation")
plt.legend()
plt.tight_layout()
plt.savefig("results/myconv_vs_baseline.png")
plt.show()