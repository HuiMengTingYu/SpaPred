import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, roc_auc_score, matthews_corrcoef, 
                            precision_score, recall_score, f1_score)  # 添加label_binarize导入
from sklearn.preprocessing import label_binarize
from SpaPred import SpaPredModel  # 确保SpaPred.py在同一目录下

class SpaPredCrossValidator:
    def __init__(self, config, n_folds=10):
        self.config = config
        self.n_folds = n_folds
        self.output_dir = "/home/zsb/Multi_model/SpaPred_10Fold"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 存储每折结果的列表
        self.all_metrics = {
            'region': {'acc': [], 'auc': [], 'mcc': [], 'pre': [], 'rec': [], 'f1': []},
            'time': {'acc': [], 'auc': [], 'mcc': [], 'pre': [], 'rec': [], 'f1': []}
        }
    
    def load_full_data(self):
        """加载完整数据集（合并训练集和测试集）"""
        # 加载训练数据
        X_train = pd.read_csv(self.config['data_paths']['X_train'], index_col=0).values
        y_region_train = pd.read_csv(self.config['data_paths']['y_region_train'])['x'].values
        y_time_train = pd.read_csv(self.config['data_paths']['y_time_train'])['x'].values
        
        # 加载测试数据
        X_test = pd.read_csv(self.config['data_paths']['X_test'], index_col=0).values
        y_region_test = pd.read_csv(self.config['data_paths']['y_region_test'])['x'].values
        y_time_test = pd.read_csv(self.config['data_paths']['y_time_test'])['x'].values
        
        # 合并数据集
        self.X_full = np.vstack((X_train, X_test))
        self.y_region_full = np.concatenate((y_region_train, y_region_test))
        self.y_time_full = np.concatenate((y_time_train, y_time_test))
        
        # 获取基因名称
        self.gene_names = pd.read_csv(self.config['data_paths']['X_train'], index_col=0).columns.tolist()
    
    def run_cross_validation(self):
        """执行10倍交叉验证"""
        # 为区域分类创建分层KFold
        region_kfold = StratifiedKFold(n_splits=self.n_folds, shuffle=True, 
                                      random_state=self.config['random_seed'])
        
        fold = 0
        for train_idx, test_idx in region_kfold.split(self.X_full, self.y_region_full):
            fold += 1
            print(f"\n{'='*80}")
            print(f"Starting Fold {fold}/{self.n_folds}")
            print(f"{'='*80}")
            
            # 创建fold输出目录
            fold_dir = os.path.join(self.output_dir, f"fold_{fold}")
            os.makedirs(fold_dir, exist_ok=True)
            
            # 更新配置中的输出目录
            fold_config = self.config.copy()
            fold_config['output_dir'] = fold_dir
            
            # 创建并运行模型
            spapred = SpaPredModel(fold_config)
            
            # 设置当前fold的数据
            spapred.X_train = self.X_full[train_idx]
            spapred.y_region_train = self.y_region_full[train_idx]
            spapred.y_time_train = self.y_time_full[train_idx]
            
            spapred.X_test = self.X_full[test_idx]
            spapred.y_region_test = self.y_region_full[test_idx]
            spapred.y_time_test = self.y_time_full[test_idx]
            
            spapred.gene_names = self.gene_names
            
            # 运行模型流程
            spapred.reduce_dimensions()
            spapred.train_models()
            spapred.optimize_weights()
            
            # 评估并收集指标
            region_metrics, time_metrics = self.evaluate_fold(spapred)
            
            # 存储指标
            for metric in ['acc', 'auc', 'mcc', 'pre', 'rec', 'f1']:
                self.all_metrics['region'][metric].append(region_metrics[metric])
                self.all_metrics['time'][metric].append(time_metrics[metric])
            
            print(f"\nFold {fold} completed. Region ACC: {region_metrics['acc']:.4f}, TIME ACC: {time_metrics['acc']:.4f}")
        
        # 保存所有结果
        self.save_final_results()
    
    def evaluate_fold(self, spapred):
        """评估当前fold并返回指标"""
        # 在测试集上评估模型
        spapred.region_acc, spapred.time_acc, spapred.region_pred, spapred.time_pred, spapred.region_probs, spapred.time_probs = spapred.ensemble.evaluate(
            spapred.X_test_reduced, spapred.y_region_test, spapred.y_time_test
        )
        
        # 计算区域分类指标
        region_metrics = {
            'acc': spapred.region_acc,
            'pre': precision_score(spapred.y_region_test, spapred.region_pred, average='weighted'),
            'rec': recall_score(spapred.y_region_test, spapred.region_pred, average='weighted'),
            'f1': f1_score(spapred.y_region_test, spapred.region_pred, average='weighted'),
            'mcc': matthews_corrcoef(spapred.y_region_test, spapred.region_pred)
        }
        
        # 计算时间分类指标
        time_metrics = {
            'acc': spapred.time_acc,
            'pre': precision_score(spapred.y_time_test, spapred.time_pred, average='weighted'),
            'rec': recall_score(spapred.y_time_test, spapred.time_pred, average='weighted'),
            'f1': f1_score(spapred.y_time_test, spapred.time_pred, average='weighted'),
            'mcc': matthews_corrcoef(spapred.y_time_test, spapred.time_pred)
        }
        
        # 计算AUC
        n_region_classes = len(np.unique(spapred.y_region_test))
        n_time_classes = len(np.unique(spapred.y_time_test))
        
        # 区域分类AUC
        if n_region_classes > 1:
            y_region_bin = label_binarize(spapred.y_region_test, classes=np.arange(n_region_classes))
            region_metrics['auc'] = roc_auc_score(
                y_region_bin, spapred.region_probs, multi_class='ovr', average='weighted'
            )
        else:
            region_metrics['auc'] = 1.0
        
        # 时间分类AUC
        if n_time_classes > 1:
            y_time_bin = label_binarize(spapred.y_time_test, classes=np.arange(n_time_classes))
            time_metrics['auc'] = roc_auc_score(
                y_time_bin, spapred.time_probs, multi_class='ovr', average='weighted'
            )
        else:
            time_metrics['auc'] = 1.0
        
        # 保存当前fold的指标
        fold_metrics = {
            'region': region_metrics,
            'time': time_metrics
        }
        with open(os.path.join(spapred.config['output_dir'], 'fold_metrics.json'), 'w') as f:
            json.dump(fold_metrics, f, indent=4)
        
        return region_metrics, time_metrics
    
    def save_final_results(self):
        """保存最终结果并生成报告"""
        # 计算平均指标
        final_metrics = {
            'region': {},
            'time': {}
        }
        
        for task in ['region', 'time']:
            for metric in self.all_metrics[task]:
                values = self.all_metrics[task][metric]
                final_metrics[task][f'avg_{metric}'] = np.mean(values)
                final_metrics[task][f'std_{metric}'] = np.std(values)
        
        # 保存最终指标
        with open(os.path.join(self.output_dir, 'final_metrics.json'), 'w') as f:
            json.dump(final_metrics, f, indent=4)
        
        # 生成报告PDF
        self.generate_report_pdf(final_metrics)
    
    def generate_report_pdf(self, final_metrics):
        """生成包含所有结果的PDF报告"""
        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('10-Fold Cross Validation Results', fontsize=20)
        
        # 区域分类结果
        self.plot_task_results(axes[0], self.all_metrics['region'], 'Region Classification')
        
        # 时间分类结果
        self.plot_task_results(axes[1], self.all_metrics['time'], 'TIME Classification')
        
        # 添加文本总结
        text_summary = self.create_text_summary(final_metrics)
        plt.figtext(0.5, 0.02, text_summary, ha='center', fontsize=12, wrap=True)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(os.path.join(self.output_dir, 'cross_validation_report.pdf'), dpi=300)
        plt.close()
    
    def plot_task_results(self, axes, metrics, title):
        """为特定任务绘制结果"""
        # ACC
        sns.boxplot(data=metrics['acc'], ax=axes[0])
        axes[0].set_title(f'{title} - Accuracy', fontsize=14)
        axes[0].set_ylabel('Accuracy')
        
        # AUC
        sns.boxplot(data=metrics['auc'], ax=axes[1])
        axes[1].set_title(f'{title} - AUC', fontsize=14)
        axes[1].set_ylabel('AUC')
        
        # MCC
        sns.boxplot(data=metrics['mcc'], ax=axes[2])
        axes[2].set_title(f'{title} - Matthews Correlation Coefficient', fontsize=14)
        axes[2].set_ylabel('MCC')
        
        # Precision
        sns.boxplot(data=metrics['pre'], ax=axes[3])
        axes[3].set_title(f'{title} - Precision', fontsize=14)
        axes[3].set_ylabel('Precision')
        
        # Recall
        sns.boxplot(data=metrics['rec'], ax=axes[4])
        axes[4].set_title(f'{title} - Recall', fontsize=14)
        axes[4].set_ylabel('Recall')
        
        # F1 Score
        sns.boxplot(data=metrics['f1'], ax=axes[5])
        axes[5].set_title(f'{title} - F1 Score', fontsize=14)
        axes[5].set_ylabel('F1 Score')
    
    def create_text_summary(self, final_metrics):
        """创建文本总结"""
        summary = "10-Fold Cross Validation Results Summary:\n\n"
        
        for task in ['region', 'time']:
            task_name = "Region Classification" if task == 'region' else "TIME Classification"
            summary += f"{task_name}:\n"
            
            for metric in ['acc', 'auc', 'mcc', 'pre', 'rec', 'f1']:
                avg = final_metrics[task][f'avg_{metric}']
                std = final_metrics[task][f'std_{metric}']
                summary += f"  - {metric.upper()}: {avg:.4f} ± {std:.4f}\n"
            
            summary += "\n"
        
        return summary

if __name__ == "__main__":
    # 配置参数 - 与原始模型相同的配置
    config = {
        # 数据路径
        'data_paths': {
            'X_train': "./RNA_train_x.csv",
            'y_region_train': "./Region_train_y.csv",
            'y_time_train': "./TIME_train_y.csv",
            'X_test': "./RNA_test_x.csv",
            'y_region_test': "./Region_test_y.csv",
            'y_time_test': "./TIME_test_y.csv"
        },
        'pca': {'auto': False, 'n_components': 50},
        'validation': {'val_size': 0.2},
        'class_weights': {
            'region': [0.7, 0.6, 2],
            'time': [0.8, 1.5, 0.6, 0.5, 2]
        },
        'auc_threshold': 0.98,
        'models_to_train': ['cnn', 'dnn', 'rf', 'xgb', 'gbdt', 'knn'],
        'model_params': {
            'cnn': {
                'conv_channels': [128, 256],
                'kernel_sizes': [3, 3],
                'fc_sizes': [512, 256],
                'dropout_rate': 0.3,
                'lr': 0.000015,
                'epochs': 100,
                'batch_size': 32,
                'weight_decay': 5e-4,
                'patience': 20,
                'lr_scheduler_step_size': 30,
                'lr_scheduler_gamma': 0.5
            },
            'dnn': {
                'hidden_sizes': [1024, 512, 256],
                'dropout_rate': 0.45,
                'lr': 0.000025,
                'epochs': 100,
                'batch_size': 64,
                'weight_decay': 1e-3,
                'patience': 20,
                'lr_scheduler_step_size': 30,
                'lr_scheduler_gamma': 0.5
            },
            'rf': {
                'n_estimators': 156,
                'max_depth': 8,
                'min_samples_split': 24,
                'min_samples_leaf': 10
            },
            'xgb': {
                'n_estimators': 128,
                'max_depth': 3,
                'learning_rate': 0.05,
                'subsample': 0.55,
                'colsample_bytree': 0.55,
                'reg_alpha': 0.65,
                'reg_lambda': 1.2,
                'gamma': 0.1
            },
            'gbdt': {
                'n_estimators': 128,
                'learning_rate': 0.05,
                'max_depth': 3,
                'min_samples_split': 32,
                'min_samples_leaf': 16,
                'subsample': 0.7,
                'max_features': 'sqrt',
                'reg_alpha': 0.5,
                'reg_lambda': 0.8
            },
            'knn': {
                'n_neighbors': 42,
                'weights': 'uniform',
                'algorithm': 'auto',
                'leaf_size': 24,
                'p': 2
            }
        },
        'ensemble': {'optimization_iterations': 200},
        'interpretability': {'shap_enabled': False, 'shap_samples': 256},
        'random_seed': 110,
        'output_dir': "/home/zsb/Multi_model/SpaPred",
        'skip_standardization': True
    }

    # 设置工作目录
    os.chdir("/home/zsb/Multi_model/data")
    
    # 创建并运行交叉验证
    validator = SpaPredCrossValidator(config, n_folds=10)
    validator.load_full_data()
    validator.run_cross_validation()
    
    print("\n10-fold cross validation completed!")
    print(f"Results saved to: {validator.output_dir}")
