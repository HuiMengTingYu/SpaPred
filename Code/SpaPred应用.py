import os
import pandas as pd
import numpy as np
import torch
from SpaPred import SpaPredModel  # 确保SpaPred.py在同一目录下

def predict_external_data(model_path, data_dir, output_dir):
    """
    使用训练好的SpaPred模型预测外部数据
    
    参数:
        model_path: 训练好的模型路径 (SpaPred_model.pkl)
        data_dir: 外部数据目录 (包含CSV文件)
        output_dir: 结果输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型 (使用GPU)
    print(f"Loading model from {model_path}...")
    spapred = SpaPredModel.load(model_path)
    spapred.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Model loaded. Using device: {spapred.device}")
    
    # 获取外部数据文件列表
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    print(f"Found {len(data_files)} CSV files in {data_dir}")
    
    # 处理每个数据文件
    for file_name in data_files:
        file_path = os.path.join(data_dir, file_name)
        print(f"\nProcessing: {file_name}")
        
        try:
            # 读取数据
            df = pd.read_csv(file_path, index_col=0)
            print(f"Data shape: {df.shape}")
            
            # 数据对齐处理
            aligned_df = align_data(df, spapred.gene_names)
            
            # 进行预测
            region_pred, time_pred = spapred.predict(aligned_df.values)
            
            # 映射预测结果到标签
            region_labels = ['Leading Edge', 'Transition', 'Tumor Core']
            time_labels = ['TIME-IK', 'TIME-IR', 'TIME-IC', 'TIME-IS', 'TIME-ID']
            
            region_results = [region_labels[p] for p in region_pred]
            time_results = [time_labels[p] for p in time_pred]
            
            # 创建结果DataFrame
            results_df = pd.DataFrame({
                'Sample_ID': aligned_df.index,
                'Region_Prediction': region_results,
                'TIME_Prediction': time_results,
                'Region_Code': region_pred,
                'TIME_Code': time_pred
            })
            
            # 保存结果
            output_path = os.path.join(output_dir, f"pred_{file_name}")
            results_df.to_csv(output_path, index=False)
            print(f"Predictions saved to: {output_path}")
            
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")

def align_data(external_df, expected_genes):
    """
    对齐外部数据与模型期望的基因特征
    
    参数:
        external_df: 外部数据DataFrame (索引为样本ID，列为基因)
        expected_genes: 模型期望的基因列表
        
    返回:
        对齐后的DataFrame
    """
    # 检查并添加缺失基因 (填充0)
    missing_genes = set(expected_genes) - set(external_df.columns)
    if missing_genes:
        print(f"  Adding {len(missing_genes)} missing genes (filled with 0)")
        for gene in missing_genes:
            external_df[gene] = 0
    
    # 移除多余基因
    extra_genes = set(external_df.columns) - set(expected_genes)
    if extra_genes:
        print(f"  Removing {len(extra_genes)} extra genes")
        external_df = external_df.drop(columns=list(extra_genes))
    
    # 按正确顺序排列基因
    aligned_df = external_df[expected_genes]
    return aligned_df

if __name__ == "__main__":
    # 配置路径 (根据你的实际路径修改)
    model_path = "/home/zsb/Multi_model/SpaPred/SpaPred_model.pkl"  # 训练好的模型
    external_data_dir = "/home/zsb/Multi_model/External_data"      # 外部数据目录
    output_dir = "/home/zsb/Multi_model/SpaPred_Prediction" # 结果输出目录
    
    # 运行预测
    predict_external_data(model_path, external_data_dir, output_dir)
    print("\nAll predictions completed!")