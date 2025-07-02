from sklearn.neighbors import KNeighborsClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, PrecisionRecallDisplay, f1_score, precision_score, recall_score, matthews_corrcoef
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
import shap
import json
import time
from tqdm import tqdm
import warnings
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
import joblib
import logging
from datetime import datetime
import dill
import re
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import WeightedRandomSampler

# 忽略警告
warnings.filterwarnings('ignore')

# 添加恒等变换器类
class IdentityScaler:
    """恒等变换器，不改变输入数据"""
    def __init__(self):
        self.mean_ = 0
        self.scale_ = 1
        
    def fit(self, X):
        return self
    
    def transform(self, X):
        return X
    
    def fit_transform(self, X):
        return X

class SpaPredModel:
    def __init__(self, config=None):
        """
        初始化SpaPred模型
        :param config: 包含所有模型参数的字典（可选）
        """
        if config:
            self.config = config
            self.setup_logger()
            self.set_random_seeds()
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"Using device: {self.device}")
            
            # 创建输出目录
            os.makedirs(config['output_dir'], exist_ok=True)
            self.logger.info(f"Output directory: {config['output_dir']}")
        else:
            # 预测模式不需要初始化日志等
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def setup_logger(self):
        """设置日志记录器"""
        self.logger = logging.getLogger('SpaPred')
        self.logger.setLevel(logging.INFO)
        
        # 创建带时间戳的日志文件
        timestamp_log_file = os.path.join(
            self.config['output_dir'], 
            f"spapred_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler = logging.FileHandler(timestamp_log_file)
        file_handler.setLevel(logging.INFO)
        
        # 创建固定文件名的日志文件
        fixed_log_file = os.path.join(self.config['output_dir'], "Log.txt")
        fixed_file_handler = logging.FileHandler(fixed_log_file, mode='w')
        fixed_file_handler.setLevel(logging.INFO)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 设置日志格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        fixed_file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(fixed_file_handler)
        self.logger.addHandler(console_handler)
    
    def set_random_seeds(self):
        """设置随机种子以确保结果可复现"""
        seed = self.config.get('random_seed', 100)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.logger.info(f"Set random seed to: {seed}")
    
    def load_data(self):
        """加载训练和测试数据"""
        self.logger.info("\n" + "="*50)
        self.logger.info("Loading data...")
        
        # 数据导入
        self.X_train = pd.read_csv(self.config['data_paths']['X_train'], index_col=0).values
        self.y_region_train = pd.read_csv(self.config['data_paths']['y_region_train'])['x'].values
        self.y_time_train = pd.read_csv(self.config['data_paths']['y_time_train'])['x'].values

        self.X_test = pd.read_csv(self.config['data_paths']['X_test'], index_col=0).values
        self.y_region_test = pd.read_csv(self.config['data_paths']['y_region_test'])['x'].values
        self.y_time_test = pd.read_csv(self.config['data_paths']['y_time_test'])['x'].values
        
        # 获取基因名称
        self.gene_names = pd.read_csv(self.config['data_paths']['X_train'], index_col=0).columns.tolist()
        
        self.logger.info(f"Training data shape: {self.X_train.shape}")
        self.logger.info(f"Test data shape: {self.X_test.shape}")
    
    def reduce_dimensions(self):
        """使用PCA降维，保留前50个主成分或自动选择"""
        self.logger.info("\n" + "="*50)
        self.logger.info("Reducing dimensions with PCA...")
        
        # 根据配置选择PCA模式
        pca_config = self.config.get('pca', {'auto': False, 'n_components': 50})
        auto_pca = pca_config.get('auto', False)
        manual_components = pca_config.get('n_components', 50)
        
        if auto_pca:
            self.logger.info("Using automatic PCA component selection")
            # 先使用全部分量拟合以计算累计方差
            pca_full = PCA(random_state=self.config['random_seed'])
            X_train_scaled = StandardScaler().fit_transform(self.X_train)
            pca_full.fit(X_train_scaled)
            
            # 计算累计方差
            cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
            # 找到达到95%方差的组件数
            n_components = np.argmax(cumulative_variance >= 0.95) + 1
            self.logger.info(f"Auto-selected {n_components} components to capture 95% variance")
        else:
            n_components = manual_components
            self.logger.info(f"Using manual PCA with {n_components} components")
        
        # 初始化并拟合PCA
        self.pca = PCA(n_components=n_components, random_state=self.config['random_seed'])
        
        # 检查是否需要跳过标准化
        skip_standardization = self.config.get('skip_standardization', False)
        
        if skip_standardization:
            self.logger.info("Skipping standardization - using pre-scaled data")
            self.scaler = IdentityScaler()
            X_train_scaled = self.X_train
            X_test_scaled = self.X_test
        else:
            # 标准化数据
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(self.X_train)
            X_test_scaled = self.scaler.transform(self.X_test)
        
        # 应用PCA
        self.X_train_reduced = self.pca.fit_transform(X_train_scaled)
        
        # 转换测试数据
        self.X_test_reduced = self.pca.transform(X_test_scaled)
        
        self.logger.info(f"Data reduced from {self.X_train.shape[1]} to {self.X_train_reduced.shape[1]} dimensions")
        self.logger.info(f"Explained variance ratio: {sum(self.pca.explained_variance_ratio_):.4f}")
        
        # 保存降维后的数据
        np.save(f"{self.config['output_dir']}/X_train_reduced.npy", self.X_train_reduced)
        np.save(f"{self.config['output_dir']}/X_test_reduced.npy", self.X_test_reduced)
        
        # 保存PCA模型和scaler
        joblib.dump(self.pca, f"{self.config['output_dir']}/pca_model.pkl")
        joblib.dump(self.scaler, f"{self.config['output_dir']}/scaler.pkl")
        
        # 划分验证集
        val_size = self.config['validation']['val_size']
        self.X_train_final, self.X_val, self.y_region_train_final, self.y_region_val, self.y_time_train_final, self.y_time_val = train_test_split(
            self.X_train_reduced, self.y_region_train, self.y_time_train, 
            test_size=val_size, random_state=self.config['random_seed'], stratify=self.y_region_train  # 添加分层抽样
        )
        
        self.logger.info(f"Final training set size: {self.X_train_final.shape[0]}")
        self.logger.info(f"Validation set size: {self.X_val.shape[0]}")
    
    def train_models(self):
        """训练所有基础模型"""
        self.logger.info("\n" + "="*50)
        self.logger.info("Training base models...")
        
        self.ensemble = WeightedModelEnsemble(
            region_classes=len(np.unique(self.y_region_train)),
            time_classes=len(np.unique(self.y_time_train)),
            device=self.device
        )
        
        # 获取类别权重并转换为张量
        region_class_weights = self.config.get('class_weights', {}).get('region', None)
        time_class_weights = self.config.get('class_weights', {}).get('time', None)
        
        # 转换为PyTorch张量
        if region_class_weights is not None:
            region_class_weights = torch.tensor(region_class_weights, dtype=torch.float32).to(self.device)
        
        if time_class_weights is not None:
            time_class_weights = torch.tensor(time_class_weights, dtype=torch.float32).to(self.device)
        
        # 初始化存储测试性能的字典
        self.base_model_test_perf = {
            'region': {},
            'time': {}
        }
        
        # 训练配置中的每种模型
        for model_type in self.config['models_to_train']:
            self.logger.info("\n" + "-"*50)
            self.logger.info(f"Training {model_type.upper()} model...")
            
            if model_type in self.config['model_params']:
                model_params = self.config['model_params'][model_type]
            else:
                model_params = {}
            # 获取类别权重
            region_model, time_model, region_weight, time_weight = train_model(
                self.X_train_final, self.y_region_train_final, self.y_time_train_final,
                self.X_val, self.y_region_val, self.y_time_val,
                input_dim=self.X_train_final.shape[1],
                model_type=model_type,
                device=self.device,
                model_params=model_params,
                logger=self.logger,
                output_dir=self.config['output_dir'],
                region_class_weights=region_class_weights,  # 传递转换后的张量
                time_class_weights=time_class_weights,      # 传递转换后的张量
                X_test=self.X_test_reduced,
                y_region_test=self.y_region_test,
                y_time_test=self.y_time_test,
                auc_threshold=self.config.get('auc_threshold', 0.95)  # 添加AUC阈值参数
            )
            
            # 添加到集成
            self.ensemble.add_model(region_model, time_model, region_weight, time_weight)
            
            # 保存模型
            if model_type == 'dnn' or model_type == 'cnn':
                torch.save(region_model.state_dict(), 
                          f"{self.config['output_dir']}/{model_type}_model.pth")
                self.logger.info(f"Saved {model_type} model to {self.config['output_dir']}/{model_type}_model.pth")
            else:
                region_model_path = f"{self.config['output_dir']}/{model_type}_region_model.pkl"
                time_model_path = f"{self.config['output_dir']}/{model_type}_time_model.pkl"
                joblib.dump(region_model, region_model_path)
                joblib.dump(time_model, time_model_path)
                self.logger.info(f"Saved {model_type} region model to {region_model_path}")
                self.logger.info(f"Saved {model_type} time model to {time_model_path}")
            
            # 在测试集上评估基础模型并存储结果
            if model_type in ['dnn', 'cnn']:
                # PyTorch模型预测
                with torch.no_grad():
                    X_tensor = torch.tensor(self.X_test_reduced, dtype=torch.float32).to(self.device)
                    region_output, time_output = region_model(X_tensor)
                    region_pred = torch.argmax(region_output, dim=1).cpu().numpy()
                    time_pred = torch.argmax(time_output, dim=1).cpu().numpy()
            else:
                # scikit-learn模型预测
                region_pred = region_model.predict(self.X_test_reduced)
                time_pred = time_model.predict(self.X_test_reduced)
            
            # 计算测试准确率
            region_acc = accuracy_score(self.y_region_test, region_pred)
            time_acc = accuracy_score(self.y_time_test, time_pred)
            
            # 存储测试性能
            self.base_model_test_perf['region'][model_type] = region_acc
            self.base_model_test_perf['time'][model_type] = time_acc
            
            self.logger.info(f"{model_type.upper()} Test Region Accuracy: {region_acc*100:.2f}%")
            self.logger.info(f"{model_type.upper()} Test TIME Accuracy: {time_acc*100:.2f}%")
            
            self.logger.info(f"{model_type.upper()} model training completed")
    
    def optimize_weights(self):
        """优化模型权重"""
        self.logger.info("\n" + "="*50)
        self.logger.info("Optimizing model weights...")
        
        self.best_region_weights, self.best_time_weights = optimize_weights(
            self.ensemble, 
            self.X_val, 
            self.y_region_val, 
            self.y_time_val,
            n_iter=self.config['ensemble']['optimization_iterations'],
            logger=self.logger
        )
        
        # 设置优化后的权重
        self.ensemble.set_weights(self.best_region_weights, self.best_time_weights)
        
        # 保存权重
        weights_data = {
            'region_weights': self.best_region_weights,
            'time_weights': self.best_time_weights,
            'model_types': self.config['models_to_train']
        }
        with open(f"{self.config['output_dir']}/ensemble_weights.json", 'w') as f:
            json.dump(weights_data, f)
    
    def evaluate(self):
        """在测试集上评估模型"""
        self.logger.info("\n" + "="*50)
        self.logger.info("Evaluating on test set...")
        
        self.region_acc, self.time_acc, self.region_pred, self.time_pred, self.region_probs, self.time_probs = self.ensemble.evaluate(
            self.X_test_reduced, self.y_region_test, self.y_time_test
        )
        
        # 计算区域分类指标
        region_precision = precision_score(self.y_region_test, self.region_pred, average='weighted')
        region_recall = recall_score(self.y_region_test, self.region_pred, average='weighted')
        region_f1 = f1_score(self.y_region_test, self.region_pred, average='weighted')
        region_mcc = matthews_corrcoef(self.y_region_test, self.region_pred)
        
        # 计算时间分类指标
        time_precision = precision_score(self.y_time_test, self.time_pred, average='weighted')
        time_recall = recall_score(self.y_time_test, self.time_pred, average='weighted')
        time_f1 = f1_score(self.y_time_test, self.time_pred, average='weighted')
        time_mcc = matthews_corrcoef(self.y_time_test, self.time_pred)
        
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Final Test Performance:")
        self.logger.info(f"Region Classification Accuracy: {self.region_acc*100:.2f}%")
        self.logger.info(f"Region Precision: {region_precision:.4f}")
        self.logger.info(f"Region Recall: {region_recall:.4f}")
        self.logger.info(f"Region F1 Score: {region_f1:.4f}")
        self.logger.info(f"Region MCC: {region_mcc:.4f}")
        
        self.logger.info(f"\nTIME Classification Accuracy: {self.time_acc*100:.2f}%")
        self.logger.info(f"TIME Precision: {time_precision:.4f}")
        self.logger.info(f"TIME Recall: {time_recall:.4f}")
        self.logger.info(f"TIME F1 Score: {time_f1:.4f}")
        self.logger.info(f"TIME MCC: {time_mcc:.4f}")
        
        # 输出分类报告
        region_report = classification_report(self.y_region_test, self.region_pred, 
                                             target_names=['Leading Edge', 'Transition', 'Tumor Core'])
        time_report = classification_report(self.y_time_test, self.time_pred,
                                           target_names=['TIME-IK', 'TIME-IR', 'TIME-IC', 'TIME-IS', 'TIME-ID'])
        
        self.logger.info("\nRegion Classification Report:")
        self.logger.info(region_report)
        
        self.logger.info("\nTIME Classification Report:")
        self.logger.info(time_report)
        
        # 保存分类报告
        with open(f"{self.config['output_dir']}/region_classification_report.txt", 'w') as f:
            f.write(region_report)
        
        with open(f"{self.config['output_dir']}/time_classification_report.txt", 'w') as f:
            f.write(time_report)
        
        # 保存所有指标
        metrics_data = {
            'region': {
                'accuracy': self.region_acc,
                'precision': region_precision,
                'recall': region_recall,
                'f1': region_f1,
                'mcc': region_mcc
            },
            'time': {
                'accuracy': self.time_acc,
                'precision': time_precision,
                'recall': time_recall,
                'f1': time_f1,
                'mcc': time_mcc
            }
        }
        with open(f"{self.config['output_dir']}/test_metrics.json", 'w') as f:
            json.dump(metrics_data, f)
    
    def visualize_base_model_performance(self):
        """为每个基础模型生成性能图（混淆矩阵、ROC曲线和精确-召回曲线）"""
        self.logger.info("\n" + "="*50)
        self.logger.info("Generating base models performance visualizations...")
        
        # 加载测试集数据（降维后的）
        X_test_reduced = self.X_test_reduced
        y_region_test = self.y_region_test
        y_time_test = self.y_time_test
        
        # 为每个模型类型生成图
        for model_type in self.config['models_to_train']:
            self.logger.info(f"\nGenerating performance plots for {model_type.upper()}...")
            
            # 加载模型
            if model_type in ['dnn', 'cnn']:
                # 加载PyTorch模型
                model_path = f"{self.config['output_dir']}/{model_type}_model.pth"
                if not os.path.exists(model_path):
                    self.logger.warning(f"Model file for {model_type} not found. Skipping.")
                    continue
                
                # 初始化模型结构
                if model_type == 'dnn':
                    model = AttentionDNN(
                        input_size=X_test_reduced.shape[1],
                        hidden_sizes=self.config['model_params']['dnn'].get('hidden_sizes', [1024, 512, 256]),
                        dropout_rate=self.config['model_params']['dnn'].get('dropout_rate', 0.6),
                        attention_heads=self.config['model_params']['dnn'].get('attention_heads', 8)
                    )
                else:  # cnn
                    model = MultiTaskCNN(
                        input_size=X_test_reduced.shape[1],
                        conv_channels=self.config['model_params']['cnn'].get('conv_channels', [64, 128]),
                        kernel_sizes=self.config['model_params']['cnn'].get('kernel_sizes', [5, 3]),
                        fc_sizes=self.config['model_params']['cnn'].get('fc_sizes', [256, 128]),
                        dropout_rate=self.config['model_params']['cnn'].get('dropout_rate', 0.6)
                    )
                
                # 加载模型权重
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.to(self.device)
                model.eval()
                
                # 将模型设置为评估模式
                region_model = model
                time_model = model
            else:
                # 加载scikit-learn模型
                region_model_path = f"{self.config['output_dir']}/{model_type}_region_model.pkl"
                time_model_path = f"{self.config['output_dir']}/{model_type}_time_model.pkl"
                
                if not (os.path.exists(region_model_path)) or not (os.path.exists(time_model_path)):
                    self.logger.warning(f"Model files for {model_type} not found. Skipping.")
                    continue
                
                # 加载区域模型
                if os.path.exists(region_model_path):
                    region_model = joblib.load(region_model_path)
                else:
                    self.logger.warning(f"Region model for {model_type} not found.")
                    continue
                
                # 加载时间模型
                if os.path.exists(time_model_path):
                    time_model = joblib.load(time_model_path)
                else:
                    self.logger.warning(f"TIME model for {model_type} not found.")
                    continue
            
            # 区域分类任务
            self.logger.info("Evaluating region classification...")
            if model_type in ['dnn', 'cnn']:
                # PyTorch模型预测
                with torch.no_grad():
                    X_tensor = torch.tensor(X_test_reduced, dtype=torch.float32).to(self.device)
                    region_output, _ = region_model(X_tensor)
                    region_pred = torch.argmax(region_output, dim=1).cpu().numpy()
                    region_probs = torch.softmax(region_output, dim=1).cpu().numpy()
            else:
                # scikit-learn模型预测
                region_pred = region_model.predict(X_test_reduced)
                region_probs = region_model.predict_proba(X_test_reduced) if hasattr(region_model, "predict_proba") else None
            
            # 计算区域分类指标
            region_acc = accuracy_score(y_region_test, region_pred)
            region_precision = precision_score(y_region_test, region_pred, average='weighted')
            region_recall = recall_score(y_region_test, region_pred, average='weighted')
            region_f1 = f1_score(y_region_test, region_pred, average='weighted')
            region_mcc = matthews_corrcoef(y_region_test, region_pred)
            
            # 绘制区域分类结果 - 拆分为单独图表
            self.plot_separate_performance(
                y_region_test, region_pred, region_probs,
                ['Leading Edge', 'Transition', 'Tumor Core'],
                f"{model_type.upper()} Region Classification",
                f"{self.config['output_dir']}/{model_type}_region",
                metrics={
                    'ACC': region_acc,
                    'PRE': region_precision,
                    'REC': region_recall,
                    'F1': region_f1,
                    'MCC': region_mcc
                }
            )
            
            # TIME分类任务
            self.logger.info("Evaluating TIME classification...")
            if model_type in ['dnn', 'cnn']:
                # PyTorch模型预测
                with torch.no_grad():
                    X_tensor = torch.tensor(X_test_reduced, dtype=torch.float32).to(self.device)
                    _, time_output = time_model(X_tensor)
                    time_pred = torch.argmax(time_output, dim=1).cpu().numpy()
                    time_probs = torch.softmax(time_output, dim=1).cpu().numpy()
            else:
                # scikit-learn模型预测
                time_pred = time_model.predict(X_test_reduced)
                time_probs = time_model.predict_proba(X_test_reduced) if hasattr(time_model, "predict_proba") else None
            
            # 计算时间分类指标
            time_acc = accuracy_score(y_time_test, time_pred)
            time_precision = precision_score(y_time_test, time_pred, average='weighted')
            time_recall = recall_score(y_time_test, time_pred, average='weighted')
            time_f1 = f1_score(y_time_test, time_pred, average='weighted')
            time_mcc = matthews_corrcoef(y_time_test, time_pred)
            
            # 绘制TIME分类结果 - 拆分为单独图表
            self.plot_separate_performance(
                y_time_test, time_pred, time_probs,
                ['TIME-IK', 'TIME-IR', 'TIME-IC', 'TIME-IS', 'TIME-ID'],
                f"{model_type.upper()} TIME Classification",
                f"{self.config['output_dir']}/{model_type}_time",
                metrics={
                    'ACC': time_acc,
                    'PRE': time_precision,
                    'REC': time_recall,
                    'F1': time_f1,
                    'MCC': time_mcc
                }
            )
    
    def visualize_results(self):
        """可视化模型性能"""
        self.logger.info("\n" + "="*50)
        self.logger.info("Generating visualizations...")
        
        # 计算区域分类指标
        region_acc = accuracy_score(self.y_region_test, self.region_pred)
        region_precision = precision_score(self.y_region_test, self.region_pred, average='weighted')
        region_recall = recall_score(self.y_region_test, self.region_pred, average='weighted')
        region_f1 = f1_score(self.y_region_test, self.region_pred, average='weighted')
        region_mcc = matthews_corrcoef(self.y_region_test, self.region_pred)
        
        # 计算时间分类指标
        time_acc = accuracy_score(self.y_time_test, self.time_pred)
        time_precision = precision_score(self.y_time_test, self.time_pred, average='weighted')
        time_recall = recall_score(self.y_time_test, self.time_pred, average='weighted')
        time_f1 = f1_score(self.y_time_test, self.time_pred, average='weighted')
        time_mcc = matthews_corrcoef(self.y_time_test, self.time_pred)
        
        # 区域分类可视化 - 拆分为单独图表
        self.plot_separate_performance(
            self.y_region_test, self.region_pred, self.region_probs,
            ['Leading Edge', 'Transition', 'Tumor Core'],
            "Region Classification",
            f"{self.config['output_dir']}/region",
            metrics={
                'ACC': region_acc,
                'PRE': region_precision,
                'REC': region_recall,
                'F1': region_f1,
                'MCC': region_mcc
            }
        )
        
        # TIME分类可视化 - 拆分为单独图表
        self.plot_separate_performance(
            self.y_time_test, self.time_pred, self.time_probs,
            ['TIME-IK', 'TIME-IR', 'TIME-IC', 'TIME-IS', 'TIME-ID'],
            "TIME Classification",
            f"{self.config['output_dir']}/time",
            metrics={
                'ACC': time_acc,
                'PRE': time_precision,
                'REC': time_recall,
                'F1': time_f1,
                'MCC': time_mcc
            }
        )
        
        # 特征重要性 - 包括所有模型
        if self.config['interpretability']['shap_enabled']:
            self.logger.info("\nGenerating SHAP explanations for all models...")
            sample_indices = np.random.choice(
                len(self.X_test_reduced), 
                min(self.config['interpretability']['shap_samples'], len(self.X_test_reduced)),
                replace=False
            )
            
            # 为所有模型生成SHAP解释
            for model_type in self.config['models_to_train']:
                self.logger.info(f"\nGenerating SHAP for {model_type.upper()}...")
                
                # 加载模型
                if model_type in ['dnn', 'cnn']:
                    # 加载PyTorch模型
                    model_path = f"{self.config['output_dir']}/{model_type}_model.pth"
                    if not os.path.exists(model_path):
                        self.logger.warning(f"Model file for {model_type} not found. Skipping.")
                        continue
                    
                    # 初始化模型结构
                    if model_type == 'dnn':
                        model = AttentionDNN(
                            input_size=self.X_test_reduced.shape[1],
                            hidden_sizes=self.config['model_params']['dnn'].get('hidden_sizes', [1024, 512, 256]),
                            dropout_rate=self.config['model_params']['dnn'].get('dropout_rate', 0.6),
                            attention_heads=self.config['model_params']['dnn'].get('attention_heads', 8)
                        )
                    else:  # cnn
                        model = MultiTaskCNN(
                            input_size=self.X_test_reduced.shape[1],
                            conv_channels=self.config['model_params']['cnn'].get('conv_channels', [64, 128]),
                            kernel_sizes=self.config['model_params']['cnn'].get('kernel_sizes', [5, 3]),
                            fc_sizes=self.config['model_params']['cnn'].get('fc_sizes', [256, 128]),
                            dropout_rate=self.config['model_params']['cnn'].get('dropout_rate', 0.6)
                        )
                    
                    # 加载模型权重
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                    model.to(self.device)
                    model.eval()
                    
                    # 解释区域分类任务
                    self.explain_model(
                        model, 
                        self.X_test_reduced, 
                        [f"PC{i+1}" for i in range(self.X_test_reduced.shape[1])],
                        ['Leading Edge', 'Transition', 'Tumor Core'],
                        f"{model_type.upper()}_Region",
                        sample_indices,
                        model_type='dnn',
                        task='region'
                    )
                    
                    # 解释时间分类任务
                    self.explain_model(
                        model, 
                        self.X_test_reduced, 
                        [f"PC{i+1}" for i in range(self.X_test_reduced.shape[1])],
                        ['TIME-IK', 'TIME-IR', 'TIME-IC', 'TIME-IS', 'TIME-ID'],
                        f"{model_type.upper()}_TIME",
                        sample_indices,
                        model_type='dnn',
                        task='time'
                    )
                else:
                    # 加载scikit-learn模型
                    if model_type == 'rf':
                        region_model_path = f"{self.config['output_dir']}/rf_region_model.pkl"
                        time_model_path = f"{self.config['output_dir']}/rf_time_model.pkl"
                    elif model_type == 'xgb':
                        region_model_path = f"{self.config['output_dir']}/xgb_region_model.pkl"
                        time_model_path = f"{self.config['output_dir']}/xgb_time_model.pkl"
                    elif model_type == 'gbdt':
                        region_model_path = f"{self.config['output_dir']}/gbdt_region_model.pkl"
                        time_model_path = f"{self.config['output_dir']}/gbdt_time_model.pkl"
                    elif model_type == 'knn':
                        region_model_path = f"{self.config['output_dir']}/knn_region_model.pkl"
                        time_model_path = f"{self.config['output_dir']}/knn_time_model.pkl"
                    else:
                        continue
                    
                    if os.path.exists(region_model_path):
                        region_model = joblib.load(region_model_path)
                        self.explain_model(
                            region_model, 
                            self.X_test_reduced, 
                            [f"PC{i+1}" for i in range(self.X_test_reduced.shape[1])],
                            ['Leading Edge', 'Transition', 'Tumor Core'],
                            f"{model_type.upper()}_Region",
                            sample_indices,
                            model_type=model_type
                        )
                    
                    if os.path.exists(time_model_path):
                        time_model = joblib.load(time_model_path)
                        self.explain_model(
                            time_model, 
                            self.X_test_reduced, 
                            [f"PC{i+1}" for i in range(self.X_test_reduced.shape[1])],
                            ['TIME-IK', 'TIME-IR', 'TIME-IC', 'TIME-IS', 'TIME-ID'],
                            f"{model_type.upper()}_TIME",
                            sample_indices,
                            model_type=model_type
                        )
            
            # 解释最终的融合模型
            self.logger.info("\nGenerating SHAP for Ensemble Model...")
            self.explain_ensemble_model(sample_indices)
        
        # 添加基础模型测试性能可视化
        self.plot_base_model_test_performance()
    
    def plot_base_model_test_performance(self):
        """绘制基础模型在测试集上的性能对比"""
        self.logger.info("\nGenerating base models test performance comparison...")
        
        # 准备数据
        model_types = list(self.base_model_test_perf['region'].keys())
        region_acc = [self.base_model_test_perf['region'][m] * 100 for m in model_types]
        time_acc = [self.base_model_test_perf['time'][m] * 100 for m in model_types]
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 设置位置和宽度
        x = np.arange(len(model_types))
        width = 0.35
        
        # 绘制柱状图
        rects1 = ax.bar(x - width/2, region_acc, width, label='Region')
        rects2 = ax.bar(x + width/2, time_acc, width, label='TIME')
        
        # 添加标签和标题
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Base Models Performance on Test Set')
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in model_types])
        ax.legend()
        
        # 添加数值标签
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        autolabel(rects1)
        autolabel(rects2)
        
        plt.tight_layout()
        
        # 保存图表
        save_path = f"{self.config['output_dir']}/base_models_test_performance.pdf"
        plt.savefig(save_path)
        plt.close()
        self.logger.info(f"Saved base models test performance to: {save_path}")
    
    def plot_metrics_table(self, metrics, title, save_path):
        """绘制性能指标表格"""
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis('off')
        ax.axis('tight')
        
        # 创建表格数据
        table_data = [
            ["ACC", f"{metrics['ACC']:.4f}"],
            ["PRE", f"{metrics['PRE']:.4f}"],
            ["REC", f"{metrics['REC']:.4f}"],
            ["F1", f"{metrics['F1']:.4f}"],
            ["MCC", f"{metrics['MCC']:.4f}"]
        ]
        
        # 创建表格
        table = ax.table(
            cellText=table_data,
            colLabels=['Metric', 'Value'],
            cellLoc='center',
            loc='center'
        )
        
        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        
        # 设置标题
        plt.title(title, fontsize=14, pad=20)
        
        # 保存表格
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    
    def plot_separate_performance(self, y_true, y_pred, y_probs, class_names, title_prefix, save_path, metrics=None):
        """可视化模型性能，拆分为单独图表"""
        # 保存指标表格
        if metrics:
            self.plot_metrics_table(metrics, f"{title_prefix} Metrics", f"{save_path}_metrics_table.pdf")
        
        # 混淆矩阵
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, 
                   yticklabels=class_names)
        plt.title(f'{title_prefix} Confusion Matrix', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{save_path}_confusion_matrix.pdf")
        plt.close()
        
        # ROC曲线（多分类） - 只有当y_probs不为None时才绘制
        if y_probs is not None and len(np.unique(y_true)) > 1:
            try:
                plt.figure(figsize=(10, 8))
                y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
                n_classes = y_true_bin.shape[1]
                
                # 检查类别数是否匹配
                if y_probs.shape[1] != n_classes:
                    self.logger.warning(f"Skipping ROC for {title_prefix}: probability shape {y_probs.shape} doesn't match number of classes {n_classes}")
                else:
                    fpr = dict()
                    tpr = dict()
                    roc_auc = dict()
                    
                    colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan'][:n_classes]
                    for i in range(n_classes):
                        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
                        roc_auc[i] = auc(fpr[i], tpr[i])
                        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                                 label=f'Class {class_names[i]} (AUC = {roc_auc[i]:.2f})')
                    
                    plt.plot([0, 1], [0, 1], 'k--', lw=2)
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate', fontsize=12)
                    plt.ylabel('True Positive Rate', fontsize=12)
                    plt.title(f'{title_prefix} ROC Curve', fontsize=14)
                    plt.legend(loc="lower right")
                    plt.tight_layout()
                    plt.savefig(f"{save_path}_roc_curve.pdf")
                    plt.close()
            except Exception as e:
                self.logger.error(f"Error generating ROC curve for {title_prefix}: {str(e)}")
        
        # 精确-召回曲线（多分类）
        if y_probs is not None and len(np.unique(y_true)) > 1:
            try:
                plt.figure(figsize=(10, 8))
                y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
                n_classes = y_true_bin.shape[1]
                
                colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan'][:n_classes]
                
                for i in range(n_classes):
                    precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
                    ap = auc(recall, precision)
                    plt.plot(recall, precision, color=colors[i], lw=2,
                             label=f'Class {class_names[i]} (AP = {ap:.2f})')
                
                plt.xlabel('Recall', fontsize=12)
                plt.ylabel('Precision', fontsize=12)
                plt.title(f'{title_prefix} Precision-Recall Curve', fontsize=14)
                plt.legend(loc="lower left")
                plt.tight_layout()
                plt.savefig(f"{save_path}_pr_curve.pdf")
                plt.close()
            except Exception as e:
                self.logger.error(f"Error generating Precision-Recall curve for {title_prefix}: {str(e)}")
    
    # 修改 explain_model 方法
    def explain_model(self, model, X, feature_names, class_names, task_name, sample_indices=None, model_type=None, task='region'):
        """使用SHAP解释模型"""
        if sample_indices is None:
            sample_indices = np.arange(len(X))
        X_sample = X[sample_indices]
        
        # 创建解释器
        if isinstance(model, nn.Module):
            # PyTorch模型解释 - 使用DeepExplainer
            if task == 'region':
                output_index = 0
            else:
                output_index = 1
                
            def model_predict(x):
                tensor_x = torch.tensor(x, dtype=torch.float32).to(self.device)
                model.eval()
                with torch.no_grad():
                    outputs = model(tensor_x)
                    # 根据任务选择正确的输出
                    if isinstance(outputs, tuple):
                        return outputs[output_index].cpu().numpy()
                    return outputs.cpu().numpy()
            
            # 使用DeepExplainer (PyTorch原生支持)
            try:
                # 使用随机样本作为背景
                background = torch.tensor(
                    X_sample[np.random.choice(X_sample.shape[0], min(10, X_sample.shape[0]), replace=False)], 
                    dtype=torch.float32
                ).to(self.device)
                
                # 创建解释器
                explainer = shap.DeepExplainer(model, background)
                
                # 计算SHAP值
                shap_values = explainer.shap_values(
                    torch.tensor(X_sample, dtype=torch.float32).to(self.device))
                
                # 绘制蜂群图
                plt.figure()
                if isinstance(shap_values, list):
                    # 多分类
                    shap.summary_plot(
                        shap_values, 
                        X_sample,
                        feature_names=feature_names, 
                        class_names=class_names,
                        show=False,
                        max_display=20
                    )
                else:
                    # 二分类
                    shap.summary_plot(
                        shap_values, 
                        X_sample,
                        feature_names=feature_names, 
                        show=False,
                        max_display=20
                    )
                plt.title(f"SHAP Summary - {task_name}", fontsize=14)
                plt.tight_layout()
                plt.savefig(f"{self.config['output_dir']}/shap_{task_name}_summary.pdf")
                plt.close()
            except Exception as e:
                self.logger.error(f"DeepExplainer failed for {task_name}: {str(e)}")
                return
        elif model_type in ['rf', 'xgb', 'gbdt'] and hasattr(model, 'predict_proba'):
            # 树模型使用TreeExplainer
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                
                # 可视化
                plt.figure()
                if isinstance(shap_values, list) and len(shap_values) > 0:
                    # 多分类
                    shap.summary_plot(
                        shap_values, 
                        X_sample,
                        feature_names=feature_names, 
                        class_names=class_names,
                        show=False,
                        max_display=20
                    )
                else:
                    # 二分类
                    shap.summary_plot(
                        shap_values, 
                        X_sample,
                        feature_names=feature_names, 
                        show=False,
                        max_display=20
                    )
                plt.title(f"SHAP Feature Importance - {task_name}", fontsize=14)
                plt.tight_layout()
                plt.savefig(f"{self.config['output_dir']}/shap_{task_name}.pdf")
                plt.close()
            except Exception as e:
                self.logger.error(f"TreeExplainer failed for {task_name}: {str(e)}")
                return
        else:
            # 其他模型使用KernelExplainer
            try:
                # 使用少量样本作为背景
                background = shap.kmeans(X_sample, min(5, len(X_sample)))
                explainer = shap.KernelExplainer(model.predict_proba, background)
                shap_values = explainer.shap_values(X_sample)
                
                # 可视化
                plt.figure()
                if isinstance(shap_values, list) and len(shap_values) > 0:
                    # 多分类
                    shap.summary_plot(
                        shap_values, 
                        X_sample,
                        feature_names=feature_names, 
                        class_names=class_names,
                        show=False,
                        max_display=20
                    )
                else:
                    # 二分类
                    shap.summary_plot(
                        shap_values, 
                        X_sample,
                        feature_names=feature_names, 
                        show=False,
                        max_display=20
                    )
                plt.title(f"SHAP Feature Importance - {task_name}", fontsize=14)
                plt.tight_layout()
                plt.savefig(f"{self.config['output_dir']}/shap_{task_name}.pdf")
                plt.close()
            except Exception as e:
                self.logger.error(f"KernelExplainer failed for {task_name}: {str(e)}")
                return
    
    def explain_ensemble_model(self, sample_indices=None):
        """解释最终的融合模型"""
        if sample_indices is None:
            sample_indices = np.arange(len(self.X_test_reduced))
        X_sample = self.X_test_reduced[sample_indices]
        
        # 定义预测函数
        def region_predict(X):
            region_pred, _, _, _ = self.ensemble.predict(X)
            return region_pred
        
        def time_predict(X):
            _, time_pred, _, _ = self.ensemble.predict(X)
            return time_pred
        
        # 解释区域分类
        try:
            # 使用少量样本作为背景
            background = shap.kmeans(X_sample, min(5, len(X_sample)))
            
            # 区域分类解释
            explainer_region = shap.KernelExplainer(region_predict, background)
            shap_values_region = explainer_region.shap_values(X_sample)
            
            plt.figure()
            shap.summary_plot(
                shap_values_region, 
                X_sample,
                feature_names=[f"PC{i+1}" for i in range(X_sample.shape[1])],
                show=False,
                max_display=20
            )
            plt.title("SHAP Feature Importance - Ensemble Region Classification", fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{self.config['output_dir']}/shap_ensemble_region.pdf")
            plt.close()
        except Exception as e:
            self.logger.error(f"SHAP for Ensemble Region failed: {str(e)}")
        
        # 解释时间分类
        try:
            # 时间分类解释
            explainer_time = shap.KernelExplainer(time_predict, background)
            shap_values_time = explainer_time.shap_values(X_sample)
            
            plt.figure()
            shap.summary_plot(
                shap_values_time, 
                X_sample,
                feature_names=[f"PC{i+1}" for i in range(X_sample.shape[1])],
                show=False,
                max_display=20
            )
            plt.title("SHAP Feature Importance - Ensemble TIME Classification", fontsize=14)
            plt.tight_layout()
            plt.savefig(f"{self.config['output_dir']}/shap_ensemble_time.pdf")
            plt.close()
        except Exception as e:
            self.logger.error(f"SHAP for Ensemble TIME failed: {str(e)}")
    
    def save_final_model(self):
        """保存最终SpaPred模型 - 使用dill处理pickle问题"""
        self.logger.info("Saving final SpaPred model...")
        
        # 保存模型组件而不是整个对象
        model_data = {
            'ensemble_models': {
                'region_models': self.ensemble.region_models,
                'time_models': self.ensemble.time_models,
                'region_weights': self.best_region_weights,
                'time_weights': self.best_time_weights
            },
            'reducer_state': {
                'scaler': self.scaler,
                'pca': self.pca,
            },
            'config': self.config,
            'gene_names': self.gene_names
        }
        
        # 使用dill保存模型
        with open(f"{self.config['output_dir']}/SpaPred_model.pkl", 'wb') as f:
            dill.dump(model_data, f)
        
        self.logger.info(f"Saved final SpaPred model to: {self.config['output_dir']}/SpaPred_model.pkl")
    
    def predict(self, X_raw):
        """
        使用训练好的SpaPred模型对新数据进行预测
        :param X_raw: 原始表达谱数据（未降维），可以是numpy数组或pandas DataFrame
        :return: region_pred（区域预测结果）, time_pred（TIME预测结果）
        """
        # 确保输入是numpy数组
        if isinstance(X_raw, pd.DataFrame):
            X_raw = X_raw.values
        elif isinstance(X_raw, pd.Series):
            X_raw = X_raw.values.reshape(1, -1)
        elif isinstance(X_raw, list):
            X_raw = np.array(X_raw)
        
        # 确保形状正确（样本数, 特征数）
        if len(X_raw.shape) == 1:
            X_raw = X_raw.reshape(1, -1)
        
        # 使用PCA降维
        if not hasattr(self, 'pca') or not hasattr(self, 'scaler'):
            self._create_reducer_from_state()
        
        # 标准化数据
        if isinstance(self.scaler, IdentityScaler):
            # 如果使用恒等变换器，则跳过标准化
            X_scaled = X_raw
        else:
            X_scaled = self.scaler.transform(X_raw)
        
        # PCA降维
        X_reduced = self.pca.transform(X_scaled)
        
        # 使用集成模型进行预测
        if not hasattr(self, 'ensemble'):
            self._create_ensemble_from_state()
        
        region_pred, time_pred, _, _ = self.ensemble.predict(X_reduced)
        
        return region_pred, time_pred
    
    def _create_reducer_from_state(self):
        """从保存的状态创建降维器"""
        if not hasattr(self, 'model_data'):
            raise RuntimeError("Model data not loaded. Please load the model first.")
        
        reducer_state = self.model_data['reducer_state']
        
        # 加载scaler和PCA
        self.scaler = reducer_state['scaler']
        self.pca = reducer_state['pca']
    
    def _create_ensemble_from_state(self):
        """从保存的状态创建集成模型"""
        if not hasattr(self, 'model_data'):
            raise RuntimeError("Model data not loaded. Please load the model first.")
        
        ensemble_data = self.model_data['ensemble_models']
        self.ensemble = WeightedModelEnsemble(
            region_classes=3,  # 固定值，根据你的分类设置
            time_classes=5,    # 固定值，根据你的分类设置
            device=self.device
        )
        
        # 添加模型和权重
        self.ensemble.region_models = ensemble_data['region_models']
        self.ensemble.time_models = ensemble_data['time_models']
        self.ensemble.region_weights = ensemble_data['region_weights']
        self.ensemble.time_weights = ensemble_data['time_weights']
    
    @classmethod
    def load(cls, model_path):
        """
        从文件加载训练好的SpaPred模型
        :param model_path: 模型文件路径
        :return: 加载好的SpaPredModel实例
        """
        with open(model_path, 'rb') as f:
            model_data = dill.load(f)
        
        # 创建实例
        instance = cls()
        instance.model_data = model_data
        instance.config = model_data['config']
        instance.gene_names = model_data['gene_names']
        instance.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 打印加载信息
        print(f"Loaded SpaPred model from: {model_path}")
        print(f"Model contains {len(model_data['ensemble_models']['region_models'])} base models")
        print(f"Gene names: {len(instance.gene_names)} genes")
        
        return instance
    
    def run(self):
        """运行整个模型流程"""
        start_time = time.time()
        
        self.load_data()
        self.reduce_dimensions()
        self.train_models()
        self.optimize_weights()
        self.evaluate()
        self.visualize_base_model_performance()  # 为每个基础模型绘制性能图
        self.visualize_results()
        self.save_final_model()  # 保存最终模型
        
        total_time = time.time() - start_time
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"All steps completed in {total_time/60:.2f} minutes")
        self.logger.info(f"Results saved to: {self.config['output_dir']}")
        
        # 关闭日志处理器
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)

# 定义数据集类
class MultiTaskDataset(Dataset):
    def __init__(self, X, y_region, y_time):
        self.X = X
        self.y_region = y_region
        self.y_time = y_time
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y_region[idx], dtype=torch.long),
            torch.tensor(self.y_time[idx], dtype=torch.long)
        )

# 自注意力模块
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
    
    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        
        # Attention mechanism
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        
        out = self.fc_out(out)
        return out

# 带自注意力的DNN模型 - 增加dropout率
class AttentionDNN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[1024, 512, 256], region_classes=3, time_classes=5, 
                 dropout_rate=0.6, attention_heads=8):  # 增加dropout率
        super(AttentionDNN, self).__init__()
        
        # 共享特征提取层
        self.shared_layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.Dropout(dropout_rate)
        )
        
        # 自注意力层
        self.attention = SelfAttention(embed_size=hidden_sizes[0], heads=attention_heads)
        
        # 共享特征提取层2
        self.shared_layer2 = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.Dropout(dropout_rate)
        )
        
        # 区域分类分支
        self.region_branch = nn.Sequential(
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[2]),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_sizes[2], region_classes)
        )
        
        # TIME分类分支
        self.time_branch = nn.Sequential(
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_sizes[2]),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_sizes[2], time_classes)
        )
    
    def forward(self, x):
        # 第一共享层
        shared_out1 = self.shared_layer1(x)
        
        # 添加序列维度用于自注意力
        shared_out1_seq = shared_out1.unsqueeze(1)
        
        # 自注意力
        attention_out = self.attention(shared_out1_seq, shared_out1_seq, shared_out1_seq)
        attention_out = attention_out.squeeze(1)
        
        # 残差连接
        attention_out = attention_out + shared_out1
        
        # 第二共享层
        shared_out2 = self.shared_layer2(attention_out)
        
        # 分支预测
        region_output = self.region_branch(shared_out2)
        time_output = self.time_branch(shared_out2)
        
        return region_output, time_output

# CNN模型 - 增加dropout率
class MultiTaskCNN(nn.Module):
    def __init__(self, input_size, conv_channels=[64, 128], kernel_sizes=[5, 3], 
                 fc_sizes=[256, 128], region_classes=3, time_classes=5, dropout_rate=0.6):  # 增加dropout率
        super(MultiTaskCNN, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, conv_channels[0], kernel_sizes[0], padding=kernel_sizes[0]//2),
            nn.ReLU(),
            nn.BatchNorm1d(conv_channels[0]),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate)  # 增加dropout
        )
        
        conv_output_size = (input_size) // 2  # 由于池化
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_sizes[1], padding=kernel_sizes[1]//2),
            nn.ReLU(),
            nn.BatchNorm1d(conv_channels[1]),
            nn.MaxPool1d(2),
            nn.Dropout(dropout_rate)  # 增加dropout
        )
        
        conv_output_size = conv_output_size // 2
        
        # 全连接层
        self.fc_input_size = conv_channels[1] * conv_output_size
        
        self.shared_fc = nn.Sequential(
            nn.Linear(self.fc_input_size, fc_sizes[0]),
            nn.ReLU(),
            nn.BatchNorm1d(fc_sizes[0]),
            nn.Dropout(dropout_rate)  # 增加dropout
        )
        
        # 区域分类分支
        self.region_branch = nn.Sequential(
            nn.Linear(fc_sizes[0], fc_sizes[1]),
            nn.ReLU(),
            nn.BatchNorm1d(fc_sizes[1]),
            nn.Dropout(dropout_rate),  # 增加dropout
            nn.Linear(fc_sizes[1], region_classes)
        )
        
        # TIME分类分支
        self.time_branch = nn.Sequential(
            nn.Linear(fc_sizes[0], fc_sizes[1]),
            nn.ReLU(),
            nn.BatchNorm1d(fc_sizes[1]),
            nn.Dropout(dropout_rate),  # 增加dropout
            nn.Linear(fc_sizes[1], time_classes)
        )
    
    def forward(self, x):
        # 添加通道维度
        x = x.unsqueeze(1)
        
        # 卷积层
        x = self.conv1(x)
        x = self.conv2(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 共享全连接层
        shared_out = self.shared_fc(x)
        
        # 分支预测
        region_output = self.region_branch(shared_out)
        time_output = self.time_branch(shared_out)
        
        return region_output, time_output

# 加权模型融合类
class WeightedModelEnsemble:
    def __init__(self, region_classes=3, time_classes=5, device='cpu'):
        self.region_models = []
        self.time_models = []
        self.region_weights = []
        self.time_weights = []
        self.region_classes = region_classes
        self.time_classes = time_classes
        self.device = device
    
    def add_model(self, region_model, time_model, region_weight=1.0, time_weight=1.0):
        self.region_models.append(region_model)
        self.time_models.append(time_model)
        self.region_weights.append(region_weight)
        self.time_weights.append(time_weight)
    
    def set_weights(self, region_weights, time_weights):
        self.region_weights = region_weights
        self.time_weights = time_weights
    
    def predict(self, X):
        region_preds = np.zeros((len(X), self.region_classes))
        time_preds = np.zeros((len(X), self.time_classes))
        
        total_region_weight = sum(self.region_weights)
        total_time_weight = sum(self.time_weights)
        
        # 区域模型预测
        for i, model in enumerate(self.region_models):
            if isinstance(model, nn.Module):
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
                    outputs = model(X_tensor)
                    
                    # 处理多任务模型的元组输出
                    if isinstance(outputs, tuple):
                        preds = outputs[0]  # 区域预测是元组的第一个元素
                    else:
                        preds = outputs
                    
                    region_preds += torch.softmax(preds, dim=1).cpu().numpy() * (self.region_weights[i] / total_region_weight)
            else:
                if hasattr(model, "predict_proba"):
                    region_preds += model.predict_proba(X) * (self.region_weights[i] / total_region_weight)
                else:
                    # 对于没有predict_proba的模型，使用one-hot编码
                    preds = model.predict(X)
                    one_hot = np.zeros((len(preds), self.region_classes))
                    one_hot[np.arange(len(preds)), preds] = 1
                    region_preds += one_hot * (self.region_weights[i] / total_region_weight)
        
        # 时间模型预测
        for i, model in enumerate(self.time_models):
            if isinstance(model, nn.Module):
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
                    outputs = model(X_tensor)
                    
                    # 处理多任务模型的元组输出
                    if isinstance(outputs, tuple):
                        preds = outputs[1]  # 时间预测是元组的第二个元素
                    else:
                        preds = outputs
                    
                    time_preds += torch.softmax(preds, dim=1).cpu().numpy() * (self.time_weights[i] / total_time_weight)
            else:
                if hasattr(model, "predict_proba"):
                    time_preds += model.predict_proba(X) * (self.time_weights[i] / total_time_weight)
                else:
                    # 对于没有predict_proba的模型，使用one-hot编码
                    preds = model.predict(X)
                    one_hot = np.zeros((len(preds), self.time_classes))
                    one_hot[np.arange(len(preds)), preds] = 1
                    time_preds += one_hot * (self.time_weights[i] / total_time_weight)
        
        # 获取最终预测
        region_final = np.argmax(region_preds, axis=1)
        time_final = np.argmax(time_preds, axis=1)
        
        return region_final, time_final, region_preds, time_preds
    
    def evaluate(self, X, y_region, y_time):
        region_pred, time_pred, region_probs, time_probs = self.predict(X)
        region_acc = accuracy_score(y_region, region_pred)
        time_acc = accuracy_score(y_time, time_pred)
        return region_acc, time_acc, region_pred, time_pred, region_probs, time_probs

# 计算类别权重
def calculate_class_weights(y, num_classes):
    """计算类别权重以处理不平衡数据"""
    class_counts = np.bincount(y)
    total_samples = len(y)
    weights = total_samples / (num_classes * class_counts)
    return torch.tensor(weights, dtype=torch.float32)

# 模型训练函数
def train_model(X_train, y_region_train, y_time_train, X_val, y_region_val, y_time_val, 
                input_dim, model_type, device, model_params={}, logger=None, output_dir='.',
                region_class_weights=None, time_class_weights=None,
                X_test=None, y_region_test=None, y_time_test=None, auc_threshold=0.95):  # 添加AUC阈值参数
    
    if logger is None:
        logger = logging.getLogger('train_model')
    
    start_time = time.time()
    
    # 用于记录训练历史
    train_loss_history = []
    val_loss_history = []
    train_region_acc_history = []
    val_region_acc_history = []
    train_time_acc_history = []
    val_time_acc_history = []
    
    if model_type == 'dnn':
        # 获取参数或使用默认值
        hidden_sizes = model_params.get('hidden_sizes', [1024, 512, 256])
        dropout_rate = model_params.get('dropout_rate', 0.6)  # 增加dropout率
        attention_heads = model_params.get('attention_heads', 8)
        lr = model_params.get('lr', 0.0001)  # 降低学习率
        epochs = model_params.get('epochs', 180)
        batch_size = model_params.get('batch_size', 64)
        weight_decay = model_params.get('weight_decay', 2e-4)  # 增加权重衰减
        patience = model_params.get('patience', 25)  # 增加早停耐心值
        lr_scheduler_step_size = model_params.get('lr_scheduler_step_size', 40)
        lr_scheduler_gamma = model_params.get('lr_scheduler_gamma', 0.7)
        
        logger.info(f"Training AttentionDNN with params: hidden_sizes={hidden_sizes}, "
              f"dropout={dropout_rate}, attention_heads={attention_heads}, lr={lr}, weight_decay={weight_decay}")
        
        # 定义模型
        model = AttentionDNN(
            input_size=input_dim,
            hidden_sizes=hidden_sizes,
            dropout_rate=dropout_rate,
            attention_heads=attention_heads
        ).to(device)
        
        # 设置类别权重
        region_criterion = nn.CrossEntropyLoss(weight=region_class_weights.to(device)) if region_class_weights is not None else nn.CrossEntropyLoss()
        time_criterion = nn.CrossEntropyLoss(weight=time_class_weights.to(device)) if time_class_weights is not None else nn.CrossEntropyLoss()
        
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma)
        
        # 创建DataLoader
        train_dataset = MultiTaskDataset(X_train, y_region_train, y_time_train)
        val_dataset = MultiTaskDataset(X_val, y_region_val, y_time_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        best_val_loss = float('inf')
        best_val_acc = 0.0
        best_epoch = 0
        epochs_no_improve = 0
        
        # AUC早停相关变量
        best_auc = 0.0
        auc_above_threshold = False
        auc_above_threshold_count = 0
        auc_patience = 5  # AUC连续达到阈值多少次才停止
        
        # 训练循环
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            region_correct = 0
            time_correct = 0
            total_samples = 0
            
            for inputs, region_labels, time_labels in train_loader:
                inputs = inputs.to(device)
                region_labels = region_labels.to(device)
                time_labels = time_labels.to(device)
                
                # 数据增强：添加高斯噪声
                noise = torch.randn_like(inputs) * 0.01
                inputs = inputs + noise
                
                optimizer.zero_grad()
                region_output, time_output = model(inputs)
                
                loss_region = region_criterion(region_output, region_labels)
                loss_time = time_criterion(time_output, time_labels)
                loss = loss_region + loss_time
                
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                
                # 计算训练准确率
                _, region_preds = torch.max(region_output, 1)
                _, time_preds = torch.max(time_output, 1)
                
                region_correct += (region_preds == region_labels).sum().item()
                time_correct += (time_preds == time_labels).sum().item()
                total_samples += region_labels.size(0)
            
            train_loss = total_loss / len(train_loader)
            train_region_acc = 100 * region_correct / total_samples
            train_time_acc = 100 * time_correct / total_samples
            
            # 记录训练历史
            train_loss_history.append(train_loss)
            train_region_acc_history.append(train_region_acc)
            train_time_acc_history.append(train_time_acc)
            
            # 验证
            model.eval()
            val_region_correct = 0
            val_time_correct = 0
            val_total_samples = 0
            val_loss = 0.0
            all_region_probs = []
            all_region_labels = []
            
            with torch.no_grad():
                for inputs, region_labels, time_labels in val_loader:
                    inputs = inputs.to(device)
                    region_labels = region_labels.to(device)
                    time_labels = time_labels.to(device)
                    
                    region_output, time_output = model(inputs)
                    
                    # 计算验证损失
                    loss_region = region_criterion(region_output, region_labels)
                    loss_time = time_criterion(time_output, time_labels)
                    val_loss += (loss_region + loss_time).item()
                    
                    # 收集概率和标签用于AUC计算
                    region_probs = torch.softmax(region_output, dim=1)
                    all_region_probs.append(region_probs.cpu().numpy())
                    all_region_labels.append(region_labels.cpu().numpy())
                    
                    _, region_preds = torch.max(region_output, 1)
                    _, time_preds = torch.max(time_output, 1)
                    
                    val_region_correct += (region_preds == region_labels).sum().item()
                    val_time_correct += (time_preds == time_labels).sum().item()
                    val_total_samples += region_labels.size(0)
            
            # 计算AUC
            all_region_probs = np.concatenate(all_region_probs, axis=0)
            all_region_labels = np.concatenate(all_region_labels, axis=0)
            
            # 计算加权AUC
            n_classes = len(np.unique(all_region_labels))
            if n_classes > 1:
                # 二值化标签
                binarized_labels = label_binarize(all_region_labels, classes=np.arange(n_classes))
                # 计算每个类别的AUC
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(binarized_labels[:, i], all_region_probs[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                # 计算加权平均AUC
                weights = np.sum(binarized_labels, axis=0) / len(all_region_labels)
                weighted_auc = np.sum([roc_auc[i] * weights[i] for i in range(n_classes)])
            else:
                weighted_auc = 1.0
            
            val_region_acc = 100 * val_region_correct / val_total_samples
            val_time_acc = 100 * val_time_correct / val_total_samples
            val_loss /= len(val_loader)
            
            # 记录验证历史
            val_loss_history.append(val_loss)
            val_region_acc_history.append(val_region_acc)
            val_time_acc_history.append(val_time_acc)
            
            # 学习率调度
            scheduler.step()
            
            # 保存最佳模型 (基于验证损失)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = (val_region_acc + val_time_acc) / 2
                best_epoch = epoch
                epochs_no_improve = 0
                best_model_state = model.state_dict().copy()
            else:
                epochs_no_improve += 1
            
            # 检查AUC是否达到阈值
            if weighted_auc >= auc_threshold:
                if not auc_above_threshold:
                    auc_above_threshold = True
                    auc_above_threshold_count = 1
                else:
                    auc_above_threshold_count += 1
            else:
                auc_above_threshold = False
                auc_above_threshold_count = 0
            
            # AUC早停检查
            if auc_above_threshold_count >= auc_patience:
                logger.info(f'AUC-based early stopping at epoch {epoch+1}: AUC reached {weighted_auc:.4f} for {auc_above_threshold_count} consecutive epochs')
                model.load_state_dict(best_model_state)
                break
            
            # 传统早停检查
            if epochs_no_improve >= patience:
                logger.info(f'Loss-based early stopping at epoch {epoch+1}')
                model.load_state_dict(best_model_state)
                break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}] | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Train Region Acc: {train_region_acc:.2f}% | "
                      f"Train Time Acc: {train_time_acc:.2f}% | "
                      f"Val Region Acc: {val_region_acc:.2f}% | "
                      f"Val Time Acc: {val_time_acc:.2f}% | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Region AUC: {weighted_auc:.4f}")
        
        # 在验证集上计算完整指标
        model.eval()
        all_region_preds = []
        all_time_preds = []
        all_region_labels = []
        all_time_labels = []
        
        with torch.no_grad():
            for inputs, region_labels, time_labels in val_loader:
                inputs = inputs.to(device)
                region_output, time_output = model(inputs)
                
                _, region_preds = torch.max(region_output, 1)
                _, time_preds = torch.max(time_output, 1)
                
                all_region_preds.extend(region_preds.cpu().numpy())
                all_time_preds.extend(time_preds.cpu().numpy())
                all_region_labels.extend(region_labels.cpu().numpy())
                all_time_labels.extend(time_labels.cpu().numpy())
        
        # 计算区域分类指标
        region_acc = accuracy_score(all_region_labels, all_region_preds)
        region_precision = precision_score(all_region_labels, all_region_preds, average='weighted')
        region_recall = recall_score(all_region_labels, all_region_preds, average='weighted')
        region_f1 = f1_score(all_region_labels, all_region_preds, average='weighted')
        region_mcc = matthews_corrcoef(all_region_labels, all_region_preds)
        
        # 计算时间分类指标
        time_acc = accuracy_score(all_time_labels, all_time_preds)
        time_precision = precision_score(all_time_labels, all_time_preds, average='weighted')
        time_recall = recall_score(all_time_labels, all_time_preds, average='weighted')
        time_f1 = f1_score(all_time_labels, all_time_preds, average='weighted')
        time_mcc = matthews_corrcoef(all_time_labels, all_time_preds)
        
        logger.info(f"Best model at epoch {best_epoch+1} with avg val acc: {best_val_acc:.2f}%")
        logger.info(f"Region: ACC={region_acc:.4f}, PRE={region_precision:.4f}, REC={region_recall:.4f}, F1={region_f1:.4f}, MCC={region_mcc:.4f}")
        logger.info(f"TIME: ACC={time_acc:.4f}, PRE={time_precision:.4f}, REC={time_recall:.4f}, F1={time_f1:.4f}, MCC={time_mcc:.4f}")
        
        # 绘制训练历史
        plot_training_history(
            train_loss_history, val_loss_history,
            train_region_acc_history, val_region_acc_history,
            train_time_acc_history, val_time_acc_history,
            model_type, output_dir
        )
        
        # 在测试集上评估
        if X_test is not None and y_region_test is not None and y_time_test is not None:
            logger.info("Evaluating on test set...")
            with torch.no_grad():
                X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
                region_output, time_output = model(X_tensor)
                region_pred = torch.argmax(region_output, dim=1).cpu().numpy()
                time_pred = torch.argmax(time_output, dim=1).cpu().numpy()
                
                region_test_acc = accuracy_score(y_region_test, region_pred)
                time_test_acc = accuracy_score(y_time_test, time_pred)
                
                logger.info(f"Test Region Accuracy: {region_test_acc*100:.2f}%")
                logger.info(f"Test TIME Accuracy: {time_test_acc*100:.2f}%")
        
        region_model = model
        time_model = model
        region_weight = region_acc
        time_weight = time_acc
    
    elif model_type == 'cnn':
        # 获取参数或使用默认值
        conv_channels = model_params.get('conv_channels', [64, 128])
        kernel_sizes = model_params.get('kernel_sizes', [3, 3])  # 使用更小的卷积核
        fc_sizes = model_params.get('fc_sizes', [256, 128])
        dropout_rate = model_params.get('dropout_rate', 0.6)  # 增加dropout率
        lr = model_params.get('lr', 0.00001)  # 降低学习率
        epochs = model_params.get('epochs', 200)  # 减少训练轮数
        batch_size = model_params.get('batch_size', 64)
        weight_decay = model_params.get('weight_decay', 1e-3)  # 大幅增加权重衰减
        patience = model_params.get('patience', 30)
        lr_scheduler_step_size = model_params.get('lr_scheduler_step_size', 50)
        lr_scheduler_gamma = model_params.get('lr_scheduler_gamma', 0.8)
        
        logger.info(f"Training CNN with params: conv_channels={conv_channels}, "
              f"kernel_sizes={kernel_sizes}, fc_sizes={fc_sizes}, dropout={dropout_rate}")
        
        # 定义模型
        model = MultiTaskCNN(
            input_size=input_dim,
            conv_channels=conv_channels,
            kernel_sizes=kernel_sizes,
            fc_sizes=fc_sizes,
            dropout_rate=dropout_rate
        ).to(device)
        
        # 设置类别权重
        region_criterion = nn.CrossEntropyLoss(weight=region_class_weights.to(device)) if region_class_weights is not None else nn.CrossEntropyLoss()
        time_criterion = nn.CrossEntropyLoss(weight=time_class_weights.to(device)) if time_class_weights is not None else nn.CrossEntropyLoss()
        
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma)
        
        # 创建DataLoader
        train_dataset = MultiTaskDataset(X_train, y_region_train, y_time_train)
        val_dataset = MultiTaskDataset(X_val, y_region_val, y_time_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        best_val_loss = float('inf')
        best_val_acc = 0.0
        best_epoch = 0
        epochs_no_improve = 0
        
        # AUC早停相关变量
        best_auc = 0.0
        auc_above_threshold = False
        auc_above_threshold_count = 0
        auc_patience = 5  # AUC连续达到阈值多少次才停止
        
        # 训练循环
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            region_correct = 0
            time_correct = 0
            total_samples = 0
            
            for inputs, region_labels, time_labels in train_loader:
                inputs = inputs.to(device)
                region_labels = region_labels.to(device)
                time_labels = time_labels.to(device)
                
                # 数据增强：添加高斯噪声
                noise = torch.randn_like(inputs) * 0.01
                inputs = inputs + noise
                
                optimizer.zero_grad()
                region_output, time_output = model(inputs)
                
                loss_region = region_criterion(region_output, region_labels)
                loss_time = time_criterion(time_output, time_labels)
                loss = loss_region + loss_time
                
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                
                # 计算训练准确率
                _, region_preds = torch.max(region_output, 1)
                _, time_preds = torch.max(time_output, 1)
                
                region_correct += (region_preds == region_labels).sum().item()
                time_correct += (time_preds == time_labels).sum().item()
                total_samples += region_labels.size(0)
            
            train_loss = total_loss / len(train_loader)
            train_region_acc = 100 * region_correct / total_samples
            train_time_acc = 100 * time_correct / total_samples
            
            # 记录训练历史
            train_loss_history.append(train_loss)
            train_region_acc_history.append(train_region_acc)
            train_time_acc_history.append(train_time_acc)
            
            # 验证
            model.eval()
            val_region_correct = 0
            val_time_correct = 0
            val_total_samples = 0
            val_loss = 0.0
            all_region_probs = []
            all_region_labels = []
            
            with torch.no_grad():
                for inputs, region_labels, time_labels in val_loader:
                    inputs = inputs.to(device)
                    region_labels = region_labels.to(device)
                    time_labels = time_labels.to(device)
                    
                    region_output, time_output = model(inputs)
                    
                    # 计算验证损失
                    loss_region = region_criterion(region_output, region_labels)
                    loss_time = time_criterion(time_output, time_labels)
                    val_loss += (loss_region + loss_time).item()
                    
                    # 收集概率和标签用于AUC计算
                    region_probs = torch.softmax(region_output, dim=1)
                    all_region_probs.append(region_probs.cpu().numpy())
                    all_region_labels.append(region_labels.cpu().numpy())
                    
                    _, region_preds = torch.max(region_output, 1)
                    _, time_preds = torch.max(time_output, 1)
                    
                    val_region_correct += (region_preds == region_labels).sum().item()
                    val_time_correct += (time_preds == time_labels).sum().item()
                    val_total_samples += region_labels.size(0)
            
            # 计算AUC
            all_region_probs = np.concatenate(all_region_probs, axis=0)
            all_region_labels = np.concatenate(all_region_labels, axis=0)
            
            # 计算加权AUC
            n_classes = len(np.unique(all_region_labels))
            if n_classes > 1:
                # 二值化标签
                binarized_labels = label_binarize(all_region_labels, classes=np.arange(n_classes))
                # 计算每个类别的AUC
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(binarized_labels[:, i], all_region_probs[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                # 计算加权平均AUC
                weights = np.sum(binarized_labels, axis=0) / len(all_region_labels)
                weighted_auc = np.sum([roc_auc[i] * weights[i] for i in range(n_classes)])
            else:
                weighted_auc = 1.0
            
            val_region_acc = 100 * val_region_correct / val_total_samples
            val_time_acc = 100 * val_time_correct / val_total_samples
            val_loss /= len(val_loader)
            
            # 记录验证历史
            val_loss_history.append(val_loss)
            val_region_acc_history.append(val_region_acc)
            val_time_acc_history.append(val_time_acc)
            
            # 学习率调度
            scheduler.step()
            
            # 保存最佳模型 (基于验证损失)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = (val_region_acc + val_time_acc) / 2
                best_epoch = epoch
                epochs_no_improve = 0
                best_model_state = model.state_dict().copy()
            else:
                epochs_no_improve += 1
            
            # 检查AUC是否达到阈值
            if weighted_auc >= auc_threshold:
                if not auc_above_threshold:
                    auc_above_threshold = True
                    auc_above_threshold_count = 1
                else:
                    auc_above_threshold_count += 1
            else:
                auc_above_threshold = False
                auc_above_threshold_count = 0
            
            # AUC早停检查
            if auc_above_threshold_count >= auc_patience:
                logger.info(f'AUC-based early stopping at epoch {epoch+1}: AUC reached {weighted_auc:.4f} for {auc_above_threshold_count} consecutive epochs')
                model.load_state_dict(best_model_state)
                break
            
            # 传统早停检查
            if epochs_no_improve >= patience:
                logger.info(f'Loss-based early stopping at epoch {epoch+1}')
                model.load_state_dict(best_model_state)
                break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}] | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Train Region Acc: {train_region_acc:.2f}% | "
                      f"Train Time Acc: {train_time_acc:.2f}% | "
                      f"Val Region Acc: {val_region_acc:.2f}% | "
                      f"Val Time Acc: {val_time_acc:.2f}% | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Region AUC: {weighted_auc:.4f}")
        
        # 在验证集上计算完整指标
        model.eval()
        all_region_preds = []
        all_time_preds = []
        all_region_labels = []
        all_time_labels = []
        
        with torch.no_grad():
            for inputs, region_labels, time_labels in val_loader:
                inputs = inputs.to(device)
                region_output, time_output = model(inputs)
                
                _, region_preds = torch.max(region_output, 1)
                _, time_preds = torch.max(time_output, 1)
                
                all_region_preds.extend(region_preds.cpu().numpy())
                all_time_preds.extend(time_preds.cpu().numpy())
                all_region_labels.extend(region_labels.cpu().numpy())
                all_time_labels.extend(time_labels.cpu().numpy())
        
        # 计算区域分类指标
        region_acc = accuracy_score(all_region_labels, all_region_preds)
        region_precision = precision_score(all_region_labels, all_region_preds, average='weighted')
        region_recall = recall_score(all_region_labels, all_region_preds, average='weighted')
        region_f1 = f1_score(all_region_labels, all_region_preds, average='weighted')
        region_mcc = matthews_corrcoef(all_region_labels, all_region_preds)
        
        # 计算时间分类指标
        time_acc = accuracy_score(all_time_labels, all_time_preds)
        time_precision = precision_score(all_time_labels, all_time_preds, average='weighted')
        time_recall = recall_score(all_time_labels, all_time_preds, average='weighted')
        time_f1 = f1_score(all_time_labels, all_time_preds, average='weighted')
        time_mcc = matthews_corrcoef(all_time_labels, all_time_preds)
        
        logger.info(f"Best model at epoch {best_epoch+1} with avg val acc: {best_val_acc:.2f}%")
        logger.info(f"Region: ACC={region_acc:.4f}, PRE={region_precision:.4f}, REC={region_recall:.4f}, F1={region_f1:.4f}, MCC={region_mcc:.4f}")
        logger.info(f"TIME: ACC={time_acc:.4f}, PRE={time_precision:.4f}, REC={time_recall:.4f}, F1={time_f1:.4f}, MCC={time_mcc:.4f}")
        
        # 绘制训练历史
        plot_training_history(
            train_loss_history, val_loss_history,
            train_region_acc_history, val_region_acc_history,
            train_time_acc_history, val_time_acc_history,
            model_type, output_dir
        )
        
        # 在测试集上评估
        if X_test is not None and y_region_test is not None and y_time_test is not None:
            logger.info("Evaluating on test set...")
            with torch.no_grad():
                X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
                region_output, time_output = model(X_tensor)
                region_pred = torch.argmax(region_output, dim=1).cpu().numpy()
                time_pred = torch.argmax(time_output, dim=1).cpu().numpy()
                
                region_test_acc = accuracy_score(y_region_test, region_pred)
                time_test_acc = accuracy_score(y_time_test, time_pred)
                
                logger.info(f"Test Region Accuracy: {region_test_acc*100:.2f}%")
                logger.info(f"Test TIME Accuracy: {time_test_acc*100:.2f}%")
        
        region_model = model
        time_model = model
        region_weight = region_acc
        time_weight = time_acc
    
    elif model_type == 'rf':
        # 获取参数或使用默认值
        n_estimators = model_params.get('n_estimators', 300)  # 减少树的数量
        max_depth = model_params.get('max_depth', 10)  # 降低最大深度
        min_samples_split = model_params.get('min_samples_split', 10)  # 增加分裂所需样本数
        min_samples_leaf = model_params.get('min_samples_leaf', 4)  # 增加叶节点最小样本数
        max_features = model_params.get('max_features', 'sqrt')
        
        logger.info(f"Training Random Forest with params: n_estimators={n_estimators}, "
              f"max_depth={max_depth}, min_samples_split={min_samples_split}")
        
        # 训练区域分类器
        region_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=True,
            random_state=100,
            n_jobs=-1
        )
        region_model.fit(X_train, y_region_train)
        
        # 训练TIME分类器
        time_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=True,
            random_state=100,
            n_jobs=-1
        )
        time_model.fit(X_train, y_time_train)
        
        # 评估验证集
        region_val_pred = region_model.predict(X_val)
        time_val_pred = time_model.predict(X_val)
        
        region_acc = accuracy_score(y_region_val, region_val_pred)
        time_acc = accuracy_score(y_time_val, time_val_pred)
        
        # 计算区域分类指标
        region_precision = precision_score(y_region_val, region_val_pred, average='weighted')
        region_recall = recall_score(y_region_val, region_val_pred, average='weighted')
        region_f1 = f1_score(y_region_val, region_val_pred, average='weighted')
        region_mcc = matthews_corrcoef(y_region_val, region_val_pred)
        
        # 计算时间分类指标
        time_precision = precision_score(y_time_val, time_val_pred, average='weighted')
        time_recall = recall_score(y_time_val, time_val_pred, average='weighted')
        time_f1 = f1_score(y_time_val, time_val_pred, average='weighted')
        time_mcc = matthews_corrcoef(y_time_val, time_val_pred)
        
        logger.info(f"Random Forest | Val Region Acc: {region_acc*100:.2f}%")
        logger.info(f"Random Forest | Val Region Precision: {region_precision:.4f}")
        logger.info(f"Random Forest | Val Region Recall: {region_recall:.4f}")
        logger.info(f"Random Forest | Val Region F1: {region_f1:.4f}")
        logger.info(f"Random Forest | Val Region MCC: {region_mcc:.4f}")
        
        logger.info(f"Random Forest | Val TIME Acc: {time_acc*100:.2f}%")
        logger.info(f"Random Forest | Val TIME Precision: {time_precision:.4f}")
        logger.info(f"Random Forest | Val TIME Recall: {time_recall:.4f}")
        logger.info(f"Random Forest | Val TIME F1: {time_f1:.4f}")
        logger.info(f"Random Forest | Val TIME MCC: {time_mcc:.4f}")
        
        # 在测试集上评估
        if X_test is not None and y_region_test is not None and y_time_test is not None:
            logger.info("Evaluating on test set...")
            region_test_pred = region_model.predict(X_test)
            time_test_pred = time_model.predict(X_test)
            
            region_test_acc = accuracy_score(y_region_test, region_test_pred)
            time_test_acc = accuracy_score(y_time_test, time_test_pred)
            
            logger.info(f"Test Region Accuracy: {region_test_acc*100:.2f}%")
            logger.info(f"Test TIME Accuracy: {time_test_acc*100:.2f}%")
        
        region_weight = region_acc
        time_weight = time_acc
    
    # XGBoost模型
    elif model_type == 'xgb':
        # 获取参数或使用默认值
        n_estimators = model_params.get('n_estimators', 200)
        max_depth = model_params.get('max_depth', 6)
        learning_rate = model_params.get('learning_rate', 0.1)
        subsample = model_params.get('subsample', 0.8)
        colsample_bytree = model_params.get('colsample_bytree', 0.8)
        reg_alpha = model_params.get('reg_alpha', 0.1)
        reg_lambda = model_params.get('reg_lambda', 1.0)
        
        logger.info(f"Training XGBoost with params: n_estimators={n_estimators}, "
              f"max_depth={max_depth}, learning_rate={learning_rate}")
        
        # 训练区域分类器
        region_model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=100,
            n_jobs=-1
        )
        region_model.fit(X_train, y_region_train)
        
        # 训练TIME分类器
        time_model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=100,
            n_jobs=-1
        )
        time_model.fit(X_train, y_time_train)
        
        # 评估验证集
        region_val_pred = region_model.predict(X_val)
        time_val_pred = time_model.predict(X_val)
        
        region_acc = accuracy_score(y_region_val, region_val_pred)
        time_acc = accuracy_score(y_time_val, time_val_pred)
        
        # 计算区域分类指标
        region_precision = precision_score(y_region_val, region_val_pred, average='weighted')
        region_recall = recall_score(y_region_val, region_val_pred, average='weighted')
        region_f1 = f1_score(y_region_val, region_val_pred, average='weighted')
        region_mcc = matthews_corrcoef(y_region_val, region_val_pred)
        
        # 计算时间分类指标
        time_precision = precision_score(y_time_val, time_val_pred, average='weighted')
        time_recall = recall_score(y_time_val, time_val_pred, average='weighted')
        time_f1 = f1_score(y_time_val, time_val_pred, average='weighted')
        time_mcc = matthews_corrcoef(y_time_val, time_val_pred)
        
        logger.info(f"XGBoost | Val Region Acc: {region_acc*100:.2f}%")
        logger.info(f"XGBoost | Val Region Precision: {region_precision:.4f}")
        logger.info(f"XGBoost | Val Region Recall: {region_recall:.4f}")
        logger.info(f"XGBoost | Val Region F1: {region_f1:.4f}")
        logger.info(f"XGBoost | Val Region MCC: {region_mcc:.4f}")
        
        logger.info(f"XGBoost | Val TIME Acc: {time_acc*100:.2f}%")
        logger.info(f"XGBoost | Val TIME Precision: {time_precision:.4f}")
        logger.info(f"XGBoost | Val TIME Recall: {time_recall:.4f}")
        logger.info(f"XGBoost | Val TIME F1: {time_f1:.4f}")
        logger.info(f"XGBoost | Val TIME MCC: {time_mcc:.4f}")
        
        # 在测试集上评估
        if X_test is not None and y_region_test is not None and y_time_test is not None:
            logger.info("Evaluating on test set...")
            region_test_pred = region_model.predict(X_test)
            time_test_pred = time_model.predict(X_test)
            
            region_test_acc = accuracy_score(y_region_test, region_test_pred)
            time_test_acc = accuracy_score(y_time_test, time_test_pred)
            
            logger.info(f"Test Region Accuracy: {region_test_acc*100:.2f}%")
            logger.info(f"Test TIME Accuracy: {time_test_acc*100:.2f}%")
        
        region_weight = region_acc
        time_weight = time_acc
    
    # 梯度提升决策树模型
    elif model_type == 'gbdt':
        # 获取参数或使用默认值
        n_estimators = model_params.get('n_estimators', 200)
        learning_rate = model_params.get('learning_rate', 0.1)
        max_depth = model_params.get('max_depth', 5)
        min_samples_split = model_params.get('min_samples_split', 10)
        min_samples_leaf = model_params.get('min_samples_leaf', 4)
        subsample = model_params.get('subsample', 0.8)
        max_features = model_params.get('max_features', 'sqrt')
        
        logger.info(f"Training GBDT with params: n_estimators={n_estimators}, "
              f"learning_rate={learning_rate}, max_depth={max_depth}")
        
        # 训练区域分类器
        region_model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            max_features=max_features,
            random_state=100
        )
        region_model.fit(X_train, y_region_train)
        
        # 训练TIME分类器
        time_model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            max_features=max_features,
            random_state=100
        )
        time_model.fit(X_train, y_time_train)
        
        # 评估验证集
        region_val_pred = region_model.predict(X_val)
        time_val_pred = time_model.predict(X_val)
        
        region_acc = accuracy_score(y_region_val, region_val_pred)
        time_acc = accuracy_score(y_time_val, time_val_pred)
        
        # 计算区域分类指标
        region_precision = precision_score(y_region_val, region_val_pred, average='weighted')
        region_recall = recall_score(y_region_val, region_val_pred, average='weighted')
        region_f1 = f1_score(y_region_val, region_val_pred, average='weighted')
        region_mcc = matthews_corrcoef(y_region_val, region_val_pred)
        
        # 计算时间分类指标
        time_precision = precision_score(y_time_val, time_val_pred, average='weighted')
        time_recall = recall_score(y_time_val, time_val_pred, average='weighted')
        time_f1 = f1_score(y_time_val, time_val_pred, average='weighted')
        time_mcc = matthews_corrcoef(y_time_val, time_val_pred)
        
        logger.info(f"GBDT | Val Region Acc: {region_acc*100:.2f}%")
        logger.info(f"GBDT | Val Region Precision: {region_precision:.4f}")
        logger.info(f"GBDT | Val Region Recall: {region_recall:.4f}")
        logger.info(f"GBDT | Val Region F1: {region_f1:.4f}")
        logger.info(f"GBDT | Val Region MCC: {region_mcc:.4f}")
        
        logger.info(f"GBDT | Val TIME Acc: {time_acc*100:.2f}%")
        logger.info(f"GBDT | Val TIME Precision: {time_precision:.4f}")
        logger.info(f"GBDT | Val TIME Recall: {time_recall:.4f}")
        logger.info(f"GBDT | Val TIME F1: {time_f1:.4f}")
        logger.info(f"GBDT | Val TIME MCC: {time_mcc:.4f}")
        
        # 在测试集上评估
        if X_test is not None and y_region_test is not None and y_time_test is not None:
            logger.info("Evaluating on test set...")
            region_test_pred = region_model.predict(X_test)
            time_test_pred = time_model.predict(X_test)
            
            region_test_acc = accuracy_score(y_region_test, region_test_pred)
            time_test_acc = accuracy_score(y_time_test, time_test_pred)
            
            logger.info(f"Test Region Accuracy: {region_test_acc*100:.2f}%")
            logger.info(f"Test TIME Accuracy: {time_test_acc*100:.2f}%")
        
        region_weight = region_acc
        time_weight = time_acc
    
    # KNN模型
    elif model_type == 'knn':
        # 获取参数或使用默认值
        n_neighbors = model_params.get('n_neighbors', 15)
        weights = model_params.get('weights', 'distance')
        algorithm = model_params.get('algorithm', 'auto')
        leaf_size = model_params.get('leaf_size', 30)
        p = model_params.get('p', 2)  # 距离度量（2为欧氏距离）
        
        logger.info(f"Training KNN with params: n_neighbors={n_neighbors}, "
              f"weights={weights}, algorithm={algorithm}")
        
        # 训练区域分类器
        region_model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            n_jobs=-1
        )
        region_model.fit(X_train, y_region_train)
        
        # 训练TIME分类器
        time_model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            n_jobs=-1
        )
        time_model.fit(X_train, y_time_train)
        
        # 评估验证集
        region_val_pred = region_model.predict(X_val)
        time_val_pred = time_model.predict(X_val)
        
        region_acc = accuracy_score(y_region_val, region_val_pred)
        time_acc = accuracy_score(y_time_val, time_val_pred)
        
        # 计算区域分类指标
        region_precision = precision_score(y_region_val, region_val_pred, average='weighted')
        region_recall = recall_score(y_region_val, region_val_pred, average='weighted')
        region_f1 = f1_score(y_region_val, region_val_pred, average='weighted')
        region_mcc = matthews_corrcoef(y_region_val, region_val_pred)
        
        # 计算时间分类指标
        time_precision = precision_score(y_time_val, time_val_pred, average='weighted')
        time_recall = recall_score(y_time_val, time_val_pred, average='weighted')
        time_f1 = f1_score(y_time_val, time_val_pred, average='weighted')
        time_mcc = matthews_corrcoef(y_time_val, time_val_pred)
        
        logger.info(f"KNN | Val Region Acc: {region_acc*100:.2f}%")
        logger.info(f"KNN | Val Region Precision: {region_precision:.4f}")
        logger.info(f"KNN | Val Region Recall: {region_recall:.4f}")
        logger.info(f"KNN | Val Region F1: {region_f1:.4f}")
        logger.info(f"KNN | Val Region MCC: {region_mcc:.4f}")
        
        logger.info(f"KNN | Val TIME Acc: {time_acc*100:.2f}%")
        logger.info(f"KNN | Val TIME Precision: {time_precision:.4f}")
        logger.info(f"KNN | Val TIME Recall: {time_recall:.4f}")
        logger.info(f"KNN | Val TIME F1: {time_f1:.4f}")
        logger.info(f"KNN | Val TIME MCC: {time_mcc:.4f}")
        
        # 在测试集上评估
        if X_test is not None and y_region_test is not None and y_time_test is not None:
            logger.info("Evaluating on test set...")
            region_test_pred = region_model.predict(X_test)
            time_test_pred = time_model.predict(X_test)
            
            region_test_acc = accuracy_score(y_region_test, region_test_pred)
            time_test_acc = accuracy_score(y_time_test, time_test_pred)
            
            logger.info(f"Test Region Accuracy: {region_test_acc*100:.2f}%")
            logger.info(f"Test TIME Accuracy: {time_test_acc*100:.2f}%")
        
        region_weight = region_acc
        time_weight = time_acc
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # 记录训练时间
    training_time = time.time() - start_time
    logger.info(f"{model_type.upper()} training completed in {training_time/60:.2f} minutes")
    
    return region_model, time_model, region_weight, time_weight

# 解析日志文件提取ACC数据
def parse_acc_data(log_content):
    """解析日志文件，提取Region和TIME的准确率数据"""
    # 基础模型验证集ACC
    val_acc = []
    test_acc = []
    patterns = {
        "dnn": {
            "val_region": r"Region: ACC=([\d.]+),",
            "val_time": r"TIME: ACC=([\d.]+),",
            "test_region": r"DNN Test Region Accuracy: ([\d.]+)%",
            "test_time": r"DNN Test TIME Accuracy: ([\d.]+)%"
        },
        "cnn": {
            "val_region": r"Region: ACC=([\d.]+),",
            "val_time": r"TIME: ACC=([\d.]+),",
            "test_region": r"CNN Test Region Accuracy: ([\d.]+)%",
            "test_time": r"CNN Test TIME Accuracy: ([\d.]+)%"
        },
        "rf": {
            "val_region": r"Random Forest \| Val Region Acc: ([\d.]+)%",
            "val_time": r"Random Forest \| Val TIME Acc: ([\d.]+)%",
            "test_region": r"RF Test Region Accuracy: ([\d.]+)%",
            "test_time": r"RF Test TIME Accuracy: ([\d.]+)%"
        },
        "xgb": {
            "val_region": r"XGBoost \| Val Region Acc: ([\d.]+)%",
            "val_time": r"XGBoost \| Val TIME Acc: ([\d.]+)%",
            "test_region": r"XGB Test Region Accuracy: ([\d.]+)%",
            "test_time": r"XGB Test TIME Accuracy: ([\d.]+)%"
        },
        "gbdt": {
            "val_region": r"GBDT \| Val Region Acc: ([\d.]+)%",
            "val_time": r"GBDT \| Val TIME Acc: ([\d.]+)%",
            "test_region": r"GBDT Test Region Accuracy: ([\d.]+)%",
            "test_time": r"GBDT Test TIME Accuracy: ([\d.]+)%"
        },
        "knn": {
            "val_region": r"KNN \| Val Region Acc: ([\d.]+)%",
            "val_time": r"KNN \| Val TIME Acc: ([\d.]+)%",
            "test_region": r"KNN Test Region Accuracy: ([\d.]+)%",
            "test_time": r"KNN Test TIME Accuracy: ([\d.]+)%"
        }
    }
    
    # 测试集最终ACC
    ensemble_test = re.search(
        r"Region Classification Accuracy: ([\d.]+)%\n.*TIME Classification Accuracy: ([\d.]+)%",
        log_content
    )
    
    # 提取基础模型ACC
    for model, pattern_dict in patterns.items():
        # 验证集Region ACC
        match_val_region = re.search(pattern_dict["val_region"], log_content)
        if match_val_region:
            if model in ["dnn", "cnn"]:
                acc_value = float(match_val_region.group(1)) * 100
            else:
                acc_value = float(match_val_region.group(1))
            val_acc.append({
                "Model": model.upper(),
                "ACC": acc_value,
                "Type": "Validation",
                "Task": "Region"
            })
        
        # 验证集TIME ACC
        match_val_time = re.search(pattern_dict["val_time"], log_content)
        if match_val_time:
            if model in ["dnn", "cnn"]:
                acc_value = float(match_val_time.group(1)) * 100
            else:
                acc_value = float(match_val_time.group(1))
            val_acc.append({
                "Model": model.upper(),
                "ACC": acc_value,
                "Type": "Validation",
                "Task": "TIME"
            })
        
        # 测试集Region ACC
        match_test_region = re.search(pattern_dict["test_region"], log_content)
        if match_test_region:
            test_acc.append({
                "Model": model.upper(),
                "ACC": float(match_test_region.group(1)),
                "Type": "Test",
                "Task": "Region"
            })
        
        # 测试集TIME ACC
        match_test_time = re.search(pattern_dict["test_time"], log_content)
        if match_test_time:
            test_acc.append({
                "Model": model.upper(),
                "ACC": float(match_test_time.group(1)),
                "Type": "Test",
                "Task": "TIME"
            })
    
    # 添加集成模型测试集ACC
    if ensemble_test:
        test_acc.append({
            "Model": "Ensemble",
            "ACC": float(ensemble_test.group(1)),
            "Type": "Test",
            "Task": "Region"
        })
        test_acc.append({
            "Model": "Ensemble",
            "ACC": float(ensemble_test.group(2)),
            "Type": "Test",
            "Task": "TIME"
        })
    
    # 合并所有ACC数据
    acc_df = pd.DataFrame(val_acc + test_acc)
    return acc_df

# 绘制ACC柱状图
def plot_acc_results(df, output_dir):
    """绘制Region和TIME的准确率对比图"""
    # 分离Region和TIME数据
    region_df = df[df['Task'] == 'Region']
    time_df = df[df['Task'] == 'TIME']
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 16))
    
    # 绘制Region准确率
    sns.barplot(x="Model", y="ACC", hue="Type", data=region_df, palette="viridis", ax=ax1)
    ax1.set_title('Region Classification Accuracy Comparison', fontsize=16)
    ax1.set_xlabel('Model', fontsize=14)
    ax1.set_ylabel('Accuracy (%)', fontsize=14)
    ax1.set_ylim(0, 100)
    ax1.legend(title='Dataset', loc='upper right')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加数据标签
    for p in ax1.patches:
        height = p.get_height()
        ax1.text(p.get_x() + p.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # 绘制TIME准确率
    sns.barplot(x="Model", y="ACC", hue="Type", data=time_df, palette="viridis", ax=ax2)
    ax2.set_title('TIME Classification Accuracy Comparison', fontsize=16)
    ax2.set_xlabel('Model', fontsize=14)
    ax2.set_ylabel('Accuracy (%)', fontsize=14)
    ax2.set_ylim(0, 100)
    ax2.legend(title='Dataset', loc='upper right')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加数据标签
    for p in ax2.patches:
        height = p.get_height()
        ax2.text(p.get_x() + p.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "Model_Accuracy_Comparison.pdf"), dpi=300)
    plt.close()

def plot_training_history(train_losses, val_losses, 
                          train_region_acc, val_region_acc,
                          train_time_acc, val_time_acc,
                          model_name, output_dir):
    """绘制训练和验证的损失曲线及准确率曲线"""
    # 创建包含两个子图的图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
    
    # 绘制损失曲线
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title(f'{model_name.upper()} Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制准确率曲线
    ax2.plot(epochs, train_region_acc, 'g-', label='Training Region Accuracy')
    ax2.plot(epochs, val_region_acc, 'c-', label='Validation Region Accuracy')
    ax2.plot(epochs, train_time_acc, 'm-', label='Training TIME Accuracy')
    ax2.plot(epochs, val_time_acc, 'y-', label='Validation TIME Accuracy')
    ax2.set_title(f'{model_name.upper()} Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(os.path.join(output_dir, f"{model_name}_training_history.pdf"))
    plt.close()
    
# 修改主函数以正确读取日志文件
def main(output_dir):
    """主函数，读取日志并生成性能报告"""
    # 读取日志内容
    log_file = os.path.join(output_dir, "Log.txt")
    if not os.path.exists(log_file):
        print(f"Log file not found at: {log_file}")
        return
    
    with open(log_file, "r") as f:
        log_content = f.read()
    
    # 解析数据并绘图
    acc_df = parse_acc_data(log_content)
    plot_acc_results(acc_df, output_dir)
    
    # 打印数据表格
    print("\nAccuracy Data Summary:")
    print(acc_df.to_string(index=False))
    
    # 保存数据到CSV
    csv_file = os.path.join(output_dir, "Accuracy_Results.csv")
    acc_df.to_csv(csv_file, index=False)
    print(f"\nResults saved to {csv_file} and {os.path.join(output_dir, 'Model_Accuracy_Comparison.pdf')}")

# 自动权重优化函数
def optimize_weights(ensemble, X_val, y_region_val, y_time_val, n_iter=50, logger=None):
    """使用贝叶斯优化自动调整模型权重"""
    if logger is None:
        logger = logging.getLogger('optimize_weights')
    
    # 定义搜索空间（每个模型两个权重：区域和TIME）
    n_models = len(ensemble.region_models)
    space = [Real(0.2, 0.5) for _ in range(2 * n_models)]
    
    # 目标函数
    def objective(weights):
        # 拆分权重
        region_weights = weights[:n_models]
        time_weights = weights[n_models:]
        
        # 设置权重
        ensemble.set_weights(region_weights, time_weights)
        
        # 评估性能
        region_acc, time_acc, _, _, _, _ = ensemble.evaluate(
            X_val, y_region_val, y_time_val
        )
        
        # 最大化准确率（最小化负准确率）
        return -(region_acc + time_acc)
    
    # 运行优化
    res = gp_minimize(
        objective, 
        space, 
        n_calls=n_iter,
        random_state=100,
        verbose=True
    )
    
    # 提取最佳权重
    best_weights = res.x
    best_region_weights = best_weights[:n_models]
    best_time_weights = best_weights[n_models:]
    
    logger.info(f"Optimized Region Weights: {best_region_weights}")
    logger.info(f"Optimized Time Weights: {best_time_weights}")
    
    return best_region_weights, best_time_weights

if __name__ == "__main__":
    # 配置参数 - 优化后的参数
    # 设置工作目录
    os.chdir("/home/zsb/Multi_model/data")
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

        # PCA配置
        'pca': {
            'auto': False,       # 自动选择主成分数量（保留95%方差） False
            'n_components': 50  # 手动设置时使用的主成分数量
        },

        # 验证集设置
        'validation': {
            'val_size': 0.2            
        },
        
        # 类别权重设置 - 处理样本不平衡
        'class_weights': {
            'region': [0.7, 0.6, 2],   # 区域分类的类别权重 [Leading Edge, Transition, Tumor Core]
            'time': [0.8, 1.5, 0.6, 0.5, 2]  # TIME分类的类别权重 [TIME-IK, TIME-IR, TIME-IC, TIME-IS, TIME-ID]
        },
        
        # AUC阈值设置
        'auc_threshold': 0.98,  # 当验证集AUC达到此值时停止训练
        
        # 要训练的模型列表
        'models_to_train': ['cnn', 'dnn', 'rf', 'xgb', 'gbdt', 'knn'],
    }

    # 创建并运行模型
    spapred = SpaPredModel(config)
    spapred.run()
    
    # 调用主函数并传递输出目录
    main(config['output_dir'])
