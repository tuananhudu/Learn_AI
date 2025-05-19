import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def log_confusion_matrix(cm, labels, model_name="Model", save_path=None):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f"Confusion Matrix - {model_name}")
    if not save_path:
        save_path = "confusion_matrix.png"
    plt.savefig(save_path)
    plt.close()
    return save_path
