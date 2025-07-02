import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                             roc_curve, auc, precision_recall_curve, average_precision_score,
                             f1_score)
from sklearn.preprocessing import label_binarize
import torch
import dill
from tqdm import tqdm

# 路径设置
MODEL_PATH = "/home/zsb/Multi_model/SpaPred/SpaPred_model.pkl"
OUTPUT_DIR = "/home/zsb/Multi_model/SpaPred_Performance"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载模型
print("Loading SpaPred model...")
with open(MODEL_PATH, 'rb') as f:
    model_data = dill.load(f)

# 设置工作目录和数据
os.chdir("/home/zsb/Multi_model/data")
config = model_data['config']
gene_names = model_data['gene_names']
region_classes = ['Leading Edge', 'Transition', 'Tumor Core']
time_classes = ['TIME-IK', 'TIME-IR', 'TIME-IC', 'TIME-IS', 'TIME-ID']

# 加载测试数据
print("Loading test data...")
X_test = pd.read_csv(config['data_paths']['X_test'], index_col=0).values
y_region_test = pd.read_csv(config['data_paths']['y_region_test'])['x'].values
y_time_test = pd.read_csv(config['data_paths']['y_time_test'])['x'].values

print(f"Test data loaded: X_test.shape={X_test.shape}, "
      f"y_region_test.shape={y_region_test.shape}, y_time_test.shape={y_time_test.shape}")

# 加载降维器和模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
scaler = model_data['reducer_state']['scaler']
pca = model_data['reducer_state']['pca']

# 标准化和降维
print("Reducing dimensions...")
X_test_scaled = scaler.transform(X_test)
X_test_reduced = pca.transform(X_test_scaled)

# 加载集成模型
class WeightedModelEnsemble:
    def __init__(self, region_classes=3, time_classes=5, device='cpu'):
        self.region_models = []
        self.time_models = []
        self.region_weights = []
        self.time_weights = []
        self.region_classes = region_classes
        self.time_classes = time_classes
        self.device = device
    
    def set_weights(self, region_weights, time_weights):
        self.region_weights = region_weights
        self.time_weights = time_weights
    
    def predict(self, X):
        region_preds = np.zeros((len(X), self.region_classes))
        time_preds = np.zeros((len(X), self.time_classes))
        total_region_weight = sum(self.region_weights)
        total_time_weight = sum(self.time_weights)
        
        # 区域预测
        for i, model in enumerate(self.region_models):
            if isinstance(model, torch.nn.Module):
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
                    outputs = model(X_tensor)
                    # 处理多任务模型输出
                    if isinstance(outputs, tuple):
                        region_output = outputs[0]
                    else:
                        region_output = outputs
                    region_probs = torch.softmax(region_output, dim=1).cpu().numpy()
                    region_preds += region_probs * (self.region_weights[i] / total_region_weight)
            else:
                region_probs = model.predict_proba(X)
                region_preds += region_probs * (self.region_weights[i] / total_region_weight)
        
        # 时间预测
        for i, model in enumerate(self.time_models):
            if isinstance(model, torch.nn.Module):
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
                    outputs = model(X_tensor)
                    # 处理多任务模型输出
                    if isinstance(outputs, tuple):
                        time_output = outputs[1]
                    else:
                        time_output = outputs
                    time_probs = torch.softmax(time_output, dim=1).cpu().numpy()
                    time_preds += time_probs * (self.time_weights[i] / total_time_weight)
            else:
                time_probs = model.predict_proba(X)
                time_preds += time_probs * (self.time_weights[i] / total_time_weight)
        
        region_final = np.argmax(region_preds, axis=1)
        time_final = np.argmax(time_preds, axis=1)
        return region_final, time_final, region_preds, time_preds

# 创建集成模型
print("Creating ensemble model...")
ensemble = WeightedModelEnsemble(
    region_classes=len(region_classes),
    time_classes=len(time_classes),
    device=device
)
ensemble_data = model_data['ensemble_models']
ensemble.region_models = ensemble_data['region_models']
ensemble.time_models = ensemble_data['time_models']
ensemble.region_weights = ensemble_data['region_weights']
ensemble.time_weights = ensemble_data['time_weights']

# 进行预测
print("Making predictions on test set...")
region_pred, time_pred, region_probs, time_probs = ensemble.predict(X_test_reduced)

# 计算性能指标
region_acc = accuracy_score(y_region_test, region_pred)
time_acc = accuracy_score(y_time_test, time_pred)
region_f1 = f1_score(y_region_test, region_pred, average='weighted')
time_f1 = f1_score(y_time_test, time_pred, average='weighted')

print(f"\nTest Performance:")
print(f"Region Classification Accuracy: {region_acc*100:.2f}%")
print(f"Region Classification Weighted F1: {region_f1:.4f}")
print(f"TIME Classification Accuracy: {time_acc*100:.2f}%")
print(f"TIME Classification Weighted F1: {time_f1:.4f}")

# 绘制函数定义
def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, 
               yticklabels=class_names,
               annot_kws={"size": 14})
    plt.title(f'{title} Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to: {save_path}")

def plot_roc_curve(y_true, y_probs, class_names, title, save_path):
    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
    n_classes = y_true_bin.shape[1]
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    plt.figure(figsize=(10, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'][:n_classes]
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2.5,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(f'{title} ROC Curve', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curve to: {save_path}")

def plot_precision_recall_curve(y_true, y_probs, class_names, title, save_path):
    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
    n_classes = y_true_bin.shape[1]
    
    precision = dict()
    recall = dict()
    avg_precision = dict()
    
    plt.figure(figsize=(10, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'][:n_classes]
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
        avg_precision[i] = average_precision_score(y_true_bin[:, i], y_probs[:, i])
        plt.plot(recall[i], precision[i], color=colors[i], lw=2.5,
                 label=f'{class_names[i]} (AP = {avg_precision[i]:.2f})')
    
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title(f'{title} Precision-Recall Curve', fontsize=16)
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Precision-Recall curve to: {save_path}")

def plot_class_distribution(y_true, class_names, title, save_path):
    class_counts = {name: np.sum(y_true == i) for i, name in enumerate(class_names)}
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_classes]
    counts = [item[1] for item in sorted_classes]
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(class_names)))
    bars = plt.bar(labels, counts, color=colors)
    
    plt.title(f'{title} Class Distribution', fontsize=16)
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    
    # 添加数据标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}', ha='center', va='bottom', fontsize=12)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved class distribution to: {save_path}")

def plot_performance_comparison(region_acc, time_acc, region_f1, time_f1, save_path):
    metrics = ['Accuracy', 'Weighted F1']
    region_scores = [region_acc * 100, region_f1 * 100]
    time_scores = [time_acc * 100, time_f1 * 100]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.figure(figsize=(12, 8))
    rects1 = plt.bar(x - width/2, region_scores, width, label='Region', color='#1f77b4')
    rects2 = plt.bar(x + width/2, time_scores, width, label='TIME', color='#ff7f0e')
    
    plt.ylabel('Score (%)', fontsize=14)
    plt.title('Model Performance Comparison', fontsize=16)
    plt.xticks(x, metrics, fontsize=14)
    plt.ylim(0, 100)
    plt.legend(fontsize=12)
    
    # 添加数据标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.annotate(f'{height:.1f}%',
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=12)
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved performance comparison to: {save_path}")

# 生成所有可视化结果
print("\nGenerating performance visualizations...")

# 区域分类可视化
plot_confusion_matrix(
    y_region_test, region_pred, region_classes, 
    "Region Classification", 
    os.path.join(OUTPUT_DIR, "region_confusion_matrix.pdf")
)

plot_roc_curve(
    y_region_test, region_probs, region_classes, 
    "Region Classification", 
    os.path.join(OUTPUT_DIR, "region_roc_curve.pdf")
)

plot_precision_recall_curve(
    y_region_test, region_probs, region_classes, 
    "Region Classification", 
    os.path.join(OUTPUT_DIR, "region_precision_recall_curve.pdf")
)

plot_class_distribution(
    y_region_test, region_classes, 
    "Region Classification", 
    os.path.join(OUTPUT_DIR, "region_class_distribution.pdf")
)

# TIME分类可视化
plot_confusion_matrix(
    y_time_test, time_pred, time_classes, 
    "TIME Classification", 
    os.path.join(OUTPUT_DIR, "time_confusion_matrix.pdf")
)

plot_roc_curve(
    y_time_test, time_probs, time_classes, 
    "TIME Classification", 
    os.path.join(OUTPUT_DIR, "time_roc_curve.pdf")
)

plot_precision_recall_curve(
    y_time_test, time_probs, time_classes, 
    "TIME Classification", 
    os.path.join(OUTPUT_DIR, "time_precision_recall_curve.pdf")
)

plot_class_distribution(
    y_time_test, time_classes, 
    "TIME Classification", 
    os.path.join(OUTPUT_DIR, "time_class_distribution.pdf")
)

# 性能比较
plot_performance_comparison(
    region_acc, time_acc, region_f1, time_f1,
    os.path.join(OUTPUT_DIR, "performance_comparison.pdf")
)

# 保存分类报告
print("\nSaving classification reports...")
region_report = classification_report(y_region_test, region_pred, target_names=region_classes)
time_report = classification_report(y_time_test, time_pred, target_names=time_classes)

with open(os.path.join(OUTPUT_DIR, "region_classification_report.txt"), 'w') as f:
    f.write("Region Classification Report\n")
    f.write("="*40 + "\n")
    f.write(region_report)

with open(os.path.join(OUTPUT_DIR, "time_classification_report.txt"), 'w') as f:
    f.write("TIME Classification Report\n")
    f.write("="*40 + "\n")
    f.write(time_report)

# 保存性能摘要
with open(os.path.join(OUTPUT_DIR, "performance_summary.txt"), 'w') as f:
    f.write("SpaPred Model Performance Summary\n")
    f.write("="*50 + "\n\n")
    f.write(f"Region Classification Accuracy: {region_acc*100:.2f}%\n")
    f.write(f"Region Classification Weighted F1: {region_f1:.4f}\n")
    f.write(f"TIME Classification Accuracy: {time_acc*100:.2f}%\n")
    f.write(f"TIME Classification Weighted F1: {time_f1:.4f}\n\n")
    f.write("Region Classification Report\n")
    f.write("="*40 + "\n")
    f.write(region_report + "\n\n")
    f.write("TIME Classification Report\n")
    f.write("="*40 + "\n")
    f.write(time_report)

print(f"\nAll performance visualizations saved to: {OUTPUT_DIR}")
print("="*50)
print("Performance Summary:")
print(f"Region Accuracy: {region_acc*100:.2f}% | F1: {region_f1:.4f}")
print(f"TIME Accuracy: {time_acc*100:.2f}% | F1: {time_f1:.4f}")
print("="*50)