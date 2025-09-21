# plot_learning_curve.py
import matplotlib
matplotlib.use("Agg")            # no GUI needed
import matplotlib.pyplot as plt

# K values
K = [1, 2, 3, 6]

# --- RPS (your numbers) ---
rps_f1  = [0.6897, 0.7241, 0.7241, 0.7931]
rps_p   = [0.7692, 0.8077, 0.8077, 0.8846]
rps_r   = [0.6250, 0.6562, 0.6562, 0.7188]

# --- Random (your numbers) ---
rnd_f1  = [0.7018, 0.7018, 0.7719, 0.7586]
rnd_p   = [0.8000, 0.8000, 0.8800, 0.8462]
rnd_r   = [0.6250, 0.6250, 0.6875, 0.6875]

def make_plot(y_rps, y_rnd, title, filename):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(K, y_rps, marker="o", label="RPS")
    ax.plot(K, y_rnd, marker="o", label="Random")
    ax.set_title(title)
    ax.set_xlabel("K (training examples)")
    ax.set_ylabel("Score")
    ax.set_xticks(K)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=150)

make_plot(rps_f1, rnd_f1, "Learning Curve — Micro-F1", "learning_curve_f1.png")
make_plot(rps_p,  rnd_p,  "Learning Curve — Precision", "learning_curve_precision.png")
make_plot(rps_r,  rnd_r,  "Learning Curve — Recall", "learning_curve_recall.png")

print("Saved:",
      "learning_curve_f1.png,",
      "learning_curve_precision.png,",
      "learning_curve_recall.png")
