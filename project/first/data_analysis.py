#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
电力变压器状态预测系统 - 数据分析模块
第一阶段：数据获取、预处理和探索性分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ETDataAnalyzer:
    """ETDataset数据分析器"""
    
    def __init__(self, data_path):
        """
        初始化数据分析器
        
        Args:
            data_path (str): 数据文件路径
        """
        self.data_path = data_path
        self.data = None
        self.data_info = {}
        
    def load_data(self, file_name):
        """
        加载数据文件
        
        Args:
            file_name (str): 文件名
            
        Returns:
            pd.DataFrame: 加载的数据
        """
        file_path = os.path.join(self.data_path, file_name)
        print(f"正在加载数据文件: {file_path}")
        
        try:
            # 读取CSV文件
            data = pd.read_csv(file_path)
            
            # 转换时间列
            data['date'] = pd.to_datetime(data['date'])
            
            # 设置时间索引
            data.set_index('date', inplace=True)
            
            self.data = data
            print(f"数据加载成功！数据形状: {data.shape}")
            return data
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None
    
    def basic_info(self):
        """获取数据基本信息"""
        if self.data is None:
            print("请先加载数据！")
            return
            
        print("=" * 50)
        print("数据基本信息")
        print("=" * 50)
        
        # 数据形状
        print(f"数据形状: {self.data.shape}")
        print(f"时间范围: {self.data.index.min()} 到 {self.data.index.max()}")
        print(f"时间跨度: {(self.data.index.max() - self.data.index.min()).days} 天")
        
        # 数据列信息
        print("\n数据列信息:")
        print(self.data.info())
        
        # 基本统计信息
        print("\n基本统计信息:")
        print(self.data.describe())
        
        # 数据类型
        print("\n数据类型:")
        print(self.data.dtypes)
        
        return {
            'shape': self.data.shape,
            'time_range': (self.data.index.min(), self.data.index.max()),
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'describe': self.data.describe()
        }
    
    def check_missing_values(self):
        """检查缺失值"""
        if self.data is None:
            print("请先加载数据！")
            return
            
        print("=" * 50)
        print("缺失值检查")
        print("=" * 50)
        
        missing_data = self.data.isnull().sum()
        missing_percent = (missing_data / len(self.data)) * 100
        
        missing_df = pd.DataFrame({
            '缺失值数量': missing_data,
            '缺失值百分比': missing_percent
        })
        
        print(missing_df)
        
        # 检查是否有缺失值
        if missing_data.sum() == 0:
            print("\n✅ 数据中没有缺失值！")
        else:
            print(f"\n⚠️  发现 {missing_data.sum()} 个缺失值")
            
        return missing_df
    
    def check_outliers(self, method='iqr'):
        """
        检查异常值
        
        Args:
            method (str): 异常值检测方法 ('iqr', 'zscore')
        """
        if self.data is None:
            print("请先加载数据！")
            return
            
        print("=" * 50)
        print(f"异常值检查 (方法: {method})")
        print("=" * 50)
        
        outliers_info = {}
        
        for column in self.data.columns:
            if method == 'iqr':
                # IQR方法
                Q1 = self.data[column].quantile(0.25)
                Q3 = self.data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = self.data[(self.data[column] < lower_bound) | 
                                   (self.data[column] > upper_bound)]
                
            elif method == 'zscore':
                # Z-score方法
                z_scores = np.abs((self.data[column] - self.data[column].mean()) / 
                                self.data[column].std())
                outliers = self.data[z_scores > 3]
            
            outlier_count = len(outliers)
            outlier_percent = (outlier_count / len(self.data)) * 100
            
            outliers_info[column] = {
                'count': outlier_count,
                'percent': outlier_percent,
                'outliers': outliers
            }
            
            print(f"{column}: {outlier_count} 个异常值 ({outlier_percent:.2f}%)")
        
        return outliers_info
    
    def data_quality_report(self):
        """生成数据质量报告"""
        if self.data is None:
            print("请先加载数据！")
            return
            
        print("=" * 60)
        print("数据质量报告")
        print("=" * 60)
        
        # 基本信息
        basic_info = self.basic_info()
        
        # 缺失值检查
        missing_info = self.check_missing_values()
        
        # 异常值检查
        outliers_info = self.check_outliers()
        
        # 数据质量评分
        quality_score = 100
        
        # 缺失值扣分
        missing_penalty = missing_info['缺失值百分比'].sum()
        quality_score -= missing_penalty
        
        # 异常值扣分（超过5%的列）
        for col, info in outliers_info.items():
            if info['percent'] > 5:
                quality_score -= info['percent'] * 0.5
        
        quality_score = max(0, quality_score)
        
        print(f"\n数据质量评分: {quality_score:.1f}/100")
        
        if quality_score >= 90:
            print("✅ 数据质量优秀")
        elif quality_score >= 80:
            print("⚠️  数据质量良好，需要少量处理")
        elif quality_score >= 70:
            print("⚠️  数据质量一般，需要处理")
        else:
            print("❌ 数据质量较差，需要大量处理")
        
        return {
            'basic_info': basic_info,
            'missing_info': missing_info,
            'outliers_info': outliers_info,
            'quality_score': quality_score
        }

def main():
    """主函数"""
    # 数据路径
    data_path = "../../ETDataset-main/ETDataset-main/ETT-small"
    
    # 创建分析器
    analyzer = ETDataAnalyzer(data_path)
    
    # 分析不同数据集
    datasets = ['ETTh1.csv', 'ETTh2.csv', 'ETTm1.csv', 'ETTm2.csv']
    
    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"分析数据集: {dataset}")
        print(f"{'='*80}")
        
        # 加载数据
        data = analyzer.load_data(dataset)
        
        if data is not None:
            # 生成数据质量报告
            quality_report = analyzer.data_quality_report()
            
            # 保存基本信息
            analyzer.data_info[dataset] = quality_report
            
            print(f"\n{dataset} 分析完成！")

if __name__ == "__main__":
    main()
