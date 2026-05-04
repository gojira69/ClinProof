import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set global styles
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

OUTPUT_DIR = "../figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_performance_vs_configs():
    configs = ["Z1 (Zero-Shot)", "A5 (BM25)", "B1 (Dense)", "B3 (Dense+BM25)"]
    f1_scores = [45.2, 71.4, 77.1, 82.8]

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=configs, y=f1_scores, palette="viridis")
    plt.title("BioASQ: Performance by Configuration", fontsize=14, pad=15)
    plt.ylabel("Macro-F1 (%)", fontsize=12)
    plt.ylim(0, 100)

    # Add data labels
    for i, v in enumerate(f1_scores):
        ax.text(i, v + 1.5, f"{v}%", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig1_performance_vs_configs.pdf", dpi=300)
    plt.close()


def plot_class_metrics_heatmap():
    classes = ["SUPPORTED", "REFUTED", "NEI"]
    metrics = ["Precision", "Recall", "F1-Score"]
    data = np.array([
        [0.45, 0.89, 0.59],
        [0.41, 0.12, 0.18],
        [0.00, 0.00, 0.00]
    ])

    plt.figure(figsize=(7, 5))
    ax = sns.heatmap(data, annot=True, fmt=".2f", cmap="Reds",
                     xticklabels=metrics, yticklabels=classes,
                     cbar_kws={'label': 'Score (0 to 1)'})
    plt.title("MedChangeQA: Class-wise Metrics (Config G1f)",
              fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig2_class_metrics_heatmap.pdf", dpi=300)
    plt.close()


def plot_error_distribution():
    labels = [
        "Retrieval Miss / Stale Docs (F2/F3)",
        "Ensemble Split Vote (F6)",
        "Epistemic Conservatism (F4)",
        "Atomic Decomp Error (F1)",
        "Parametric Hallucination (F5)"
    ]
    sizes = [45, 25, 15, 8, 7]
    explode = (0.05, 0, 0, 0, 0)

    plt.figure(figsize=(9, 6))
    plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            startangle=140, colors=sns.color_palette("Set2"))
    plt.title("ClinProof Aggregate Error Distribution", fontsize=14, pad=20)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig3_error_distribution.pdf", dpi=300)
    plt.close()


def plot_recency_ablation():
    alphas = ["Flat (α=0.0)", "Mild (α=0.1)",
              "Moderate (α=0.3)", "Strong (α=0.7)"]
    f1_scores = [26.5, 27.1, 28.5, 29.6]

    plt.figure(figsize=(8, 5))
    plt.plot(alphas, f1_scores, marker='o', linewidth=3,
             markersize=10, color="#d9534f")
    plt.title("MedChangeQA: Recency Weighting Impact", fontsize=14, pad=15)
    plt.ylabel("Macro-F1 (%)", fontsize=12)
    plt.ylim(20, 35)
    plt.grid(True, linestyle='--', alpha=0.7)

    for i, v in enumerate(f1_scores):
        plt.text(i, v + 0.5, f"{v}%", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig4_recency_ablation.pdf", dpi=300)
    plt.close()


def plot_ensemble_agreement():
    categories = ["Unanimous (3/3 Votes)", "Split (2/3 Votes)"]
    correct = [85, 30]
    incorrect = [15, 70]

    plt.figure(figsize=(7, 6))
    bar_width = 0.5

    p1 = plt.bar(categories, correct, width=bar_width,
                 color="#5cb85c", label="Correct")
    p2 = plt.bar(categories, incorrect, width=bar_width,
                 bottom=correct, color="#d9534f", label="Incorrect")

    plt.title("Accuracy by Ensemble Confidence", fontsize=14, pad=15)
    plt.ylabel("Percentage of Predictions", fontsize=12)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig5_ensemble_agreement.pdf", dpi=300)
    plt.close()


if __name__ == "__main__":
    print("Generating visualization suite...")
    plot_performance_vs_configs()
    plot_class_metrics_heatmap()
    plot_error_distribution()
    plot_recency_ablation()
    plot_ensemble_agreement()
    print(f"All figures saved successfully to {os.path.abspath(OUTPUT_DIR)}")
