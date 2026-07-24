# CAN BE LAUNCHED LOCALLY
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import random

from src.paths import CONFIG_PATH


def main(title, *args):
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    BUSINESS_FPR = config["business_targets"]["business_fp_target"]

    mlflow.set_tracking_uri("http://localhost:5001")
    experiment_name = "fraud_detection"
    run_ids = args

    metrics_to_plot = [
        "metrics.lgbm_train_pr_auc",
        "metrics.lgbm_train_recall",
        "metrics.lgbm_train_precision",
        "metrics.lgbm_train_f1",
        "metrics.lgbm_train_roc_auc",
        "metrics.lgbm_train_recall_at_fpr",
        "metrics.lgbm_val_pr_auc",
        "metrics.lgbm_val_recall",
        "metrics.lgbm_val_precision",
        "metrics.lgbm_val_f1",
        "metrics.lgbm_val_roc_auc",
        "metrics.lgbm_val_recall_at_fpr",
        "metrics.lgbm_test_pr_auc",
        "metrics.lgbm_test_recall",
        "metrics.lgbm_test_precision",
        "metrics.lgbm_test_f1",
        "metrics.lgbm_test_roc_auc",
        "metrics.lgbm_test_recall_at_fpr",
        "metrics.lgbm_cv_pr_auc"
    ]

    df = mlflow.search_runs(experiment_names=[experiment_name])
    df_filtered = df[df["run_id"].isin(run_ids)].copy()
    df_filtered["Run_Name"] = df_filtered["tags.mlflow.runName"].fillna(df_filtered["run_id"])

    columns_to_keep = ["Run_Name"] + metrics_to_plot
    df_filtered = df_filtered[columns_to_keep]

    df_melted = df_filtered.melt(
        id_vars=["Run_Name"],
        value_vars=metrics_to_plot,
        var_name="Metric",
        value_name="Value"
    )

    df_melted["Metric"] = df_melted["Metric"].str.replace("metrics.lgbm_", "")
    metric_names_mapping = {
        "train_pr_auc": "Train PR-AUC",
        "train_recall": "Train Recall",
        "train_precision": "Train Precision",
        "train_f1": "Train F1-Score",
        "train_roc_auc": "Train ROC-AUC",
        "train_recall_at_fpr": f"Train Recall @ FPR {int(BUSINESS_FPR*100)}%",

        "val_pr_auc": "Val PR-AUC",
        "val_recall": "Val Recall",
        "val_precision": "Val Precision",
        "val_f1": "Val F1-Score",
        "val_roc_auc": "Val ROC-AUC",
        "val_recall_at_fpr": f"Val Recall @ FPR {int(BUSINESS_FPR*100)}%",

        "test_pr_auc": "Test PR-AUC",
        "test_recall": "Test Recall",
        "test_precision": "Test Precision",
        "test_f1": "Test F1-Score",
        "test_roc_auc": "Test ROC-AUC",
        "test_recall_at_fpr": f"Test Recall @ FPR {int(BUSINESS_FPR*100)}%",

        "cv_pr_auc": "Cross-Val PR-AUC"
    }

    # Если ключа нет в словаре, fillna оставит старое значение.
    df_melted["Metric"] = df_melted["Metric"].map(metric_names_mapping).fillna(df_melted["Metric"])


    # Устанавливаем стиль
    run_names = df_melted["Run_Name"].unique()
    sns.set_theme(style="whitegrid")

    base_palette = sns.color_palette("muted", 10)
    random_palette = random.sample(base_palette, len(run_names))

    # Visualization fix
    name_to_id = {name: str(i) for i, name in enumerate(run_names)}
    df_melted["Run_ID_Short"] = df_melted["Run_Name"].map(name_to_id)

    g = sns.catplot(
        data=df_melted,
        kind="bar",
        x="Run_ID_Short",
        y="Value",
        col="Metric",
        col_wrap=6,
        hue="Run_Name",
        dodge=False,
        palette=random_palette,
        sharey=True,
        height=3.5,
        aspect=0.9,
        legend=False
    )

    for ax in g.axes.flat:
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=3, fontsize=11)

        ax.set_ylim(0, 1.1)  # fixed scale

        ax.set_xticklabels([])
        ax.tick_params(bottom=False)
        ax.set_xlabel("")

    g.set_titles("{col_name}", size=13, weight="bold")
    g.set_axis_labels("", "")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=random_palette[i])
        for i in range(len(run_names))
    ]

    g.figure.legend(
        handles,
        run_names,
        title="Run name",
        bbox_to_anchor=(0.95, 0.05),
        loc='lower right',
        fontsize=12,
        title_fontsize=14,
        borderpad=1.2,
        labelspacing=0.8,
        frameon=True
    )

    # Общий заголовок для всей картинки
    g.figure.subplots_adjust(top=0.9, bottom=0.05, right=0.98, left=0.02)  # space above for a title
    g.figure.suptitle(title, fontsize=18, weight="bold")

    plt.show()

if __name__ == "__main__":
    main("Comparison of Pipelines", "a0d134f8145a40fbbf67907dc6d63bce", "1b97778161e9417896aacb767b2a7cd5") # lgbm vs pytorch+lgbm
    main("Optima Before / After", "5601aababdf544c0acf7f302fc829b80", "1b97778161e9417896aacb767b2a7cd5") # pytorch+lgbm vs unoptimized