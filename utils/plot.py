import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", save_path=None):
    sns.set_style("dark")
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="magma",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"}
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()