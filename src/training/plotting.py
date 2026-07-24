import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import mlflow
import seaborn as sns
from src.paths import PLOTS_DIR


def plot_pr_curves(y_val, y_val_prob, run_id, title_prefix="LightGBM"):
    client = mlflow.MlflowClient()
    precision, recall, thresholds = precision_recall_curve(y_val, y_val_prob)

    pr_auc = average_precision_score(y_val, y_val_prob)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ==========================================
    # График 1: Precision & Recall vs. Threshold
    # ==========================================
    ax1.plot(thresholds, precision[:-1], label="Precision", color="#2ca02c", linewidth=2)  # all but scikit last element
    ax1.plot(thresholds, recall[:-1], label="Recall", color="#d62728", linewidth=2)  # all but scikit last element

    ax1.set_title(f"{title_prefix}: Precision & Recall vs Threshold", fontsize=14, weight="bold")
    ax1.set_xlabel("Decision Threshold", fontsize=12)
    ax1.set_ylabel("Score", fontsize=12)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="best", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.7)

    # ==========================================
    # График 2: Precision-Recall Curve (PR-AUC)
    # ==========================================
    ax2.plot(recall, precision, color="#1f77b4", linewidth=2, label=f"PR Curve (AUC = {pr_auc:.3f})")

    ax2.fill_between(recall, precision, alpha=0.2, color="#1f77b4")

    ax2.set_title(f"{title_prefix}: Precision-Recall Curve", fontsize=14, weight="bold")
    ax2.set_xlabel("Recall", fontsize=12)
    ax2.set_ylabel("Precision", fontsize=12)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="lower left", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()

    # Save to file
    save_path = PLOTS_DIR / 'pr_curves.png'
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    # Save to MLflow
    client.log_artifact(run_id, save_path, 'plots')

def plot_density(y_val, y_val_prob, run_id, business_thr, f1_thr):
    client = mlflow.MlflowClient()
    plt.figure(figsize=(10, 6))
    # Legit transactions
    sns.histplot(y_val_prob[y_val == 0], color='green', label='Legit (0)', stat="density", bins=50, alpha=0.5, kde=True)
    # Fraud transactions
    sns.histplot(y_val_prob[y_val == 1], color='red', label='Fraud (1)', stat="density", bins=50, alpha=0.5, kde=True)

    plt.axvline(business_thr, color='orange', linestyle='--', label='Best Business threshold')
    plt.axvline(f1_thr, color='blue', linestyle='--', label='Best F1 threshold')

    plt.title("Predicted Probability Distribution")
    plt.xlabel("Predicted Probability of Fraud")
    plt.ylabel("Density")
    plt.xlim(0, 1)
    plt.legend()

    # Save to file
    save_path = PLOTS_DIR / 'probability_distribution.png'
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    # Save to MLflow
    client.log_artifact(run_id, save_path, 'plots')
