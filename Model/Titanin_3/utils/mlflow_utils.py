import matplotlib.pyplot as plt
import seaborn as sns
import os

def log_confusion_matrix(cm, labels, run_id=None, save_path="confusion_matrix.png"):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path
