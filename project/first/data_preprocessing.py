#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
电力变压器状态预测系统 - 数据预处理模块
包括数据清洗、标准化、归一化等功能
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, data):
        """
        初始化预处理器
        
        Args:
            data (pd.DataFrame): 原始数据
        """
        self.original_data = data.copy()
        self.processed_data = data.copy()
        self.scalers = {}
        self.preprocessing_info = {}
    
    def handle_missing_values(self, method='mean'):
        """
        处理缺失值
        
        Args:
            method (str): 处理方法 ('mean', 'median', 'mode', 'knn', 'drop')
        """
        print(f"使用 {method} 方法处理缺失值...")
        
        missing_before = self.processed_data.isnull().sum().sum()
        
        if missing_before == 0:
            print("数据中没有缺失值，跳过处理")
            return self.processed_data
        
        if method == 'drop':
            # 删除包含缺失值的行
            self.processed_data = self.processed_data.dropna()
            
        elif method == 'mean':
            # 使用均值填充
            imputer = SimpleImputer(strategy='mean')
            self.processed_data = pd.DataFrame(
                imputer.fit_transform(self.processed_data),
                columns=self.processed_data.columns,
                index=self.processed_data.index
            )
            
        elif method == 'median':
            # 使用中位数填充
            imputer = SimpleImputer(strategy='median')
            self.processed_data = pd.DataFrame(
                imputer.fit_transform(self.processed_data),
                columns=self.processed_data.columns,
                index=self.processed_data.index
            )
            
        elif method == 'mode':
            # 使用众数填充
            imputer = SimpleImputer(strategy='most_frequent')
            self.processed_data = pd.DataFrame(
                imputer.fit_transform(self.processed_data),
                columns=self.processed_data.columns,
                index=self.processed_data.index
            )
            
        elif method == 'knn':
            # 使用KNN填充
            imputer = KNNImputer(n_neighbors=5)
            self.processed_data = pd.DataFrame(
                imputer.fit_transform(self.processed_data),
                columns=self.processed_data.columns,
                index=self.processed_data.index
            )
        
        missing_after = self.processed_data.isnull().sum().sum()
        print(f"缺失值处理完成: {missing_before} -> {missing_after}")
        
        self.preprocessing_info['missing_values'] = {
            'method': method,
            'before': missing_before,
            'after': missing_after
        }
        
        return self.processed_data
    
    def handle_outliers(self, method='iqr', columns=None):
        """
        处理异常值
        
        Args:
            method (str): 处理方法 ('iqr', 'zscore', 'isolation_forest')
            columns (list): 要处理的列名
        """
        if columns is None:
            columns = self.processed_data.columns
        
        print(f"使用 {method} 方法处理异常值...")
        
        outliers_info = {}
        
        for col in columns:
            if method == 'iqr':
                # IQR方法
                Q1 = self.processed_data[col].quantile(0.25)
                Q3 = self.processed_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # 记录异常值信息
                outliers_mask = (self.processed_data[col] < lower_bound) | \
                               (self.processed_data[col] > upper_bound)
                outliers_count = outliers_mask.sum()
                
                # 用边界值替换异常值
                self.processed_data.loc[self.processed_data[col] < lower_bound, col] = lower_bound
                self.processed_data.loc[self.processed_data[col] > upper_bound, col] = upper_bound
                
            elif method == 'zscore':
                # Z-score方法
                z_scores = np.abs((self.processed_data[col] - self.processed_data[col].mean()) / 
                                self.processed_data[col].std())
                outliers_mask = z_scores > 3
                outliers_count = outliers_mask.sum()
                
                # 用均值替换异常值
                mean_val = self.processed_data[col].mean()
                self.processed_data.loc[outliers_mask, col] = mean_val
            
            outliers_info[col] = {
                'count': outliers_count,
                'percent': (outliers_count / len(self.processed_data)) * 100
            }
            
            print(f"{col}: 处理了 {outliers_count} 个异常值")
        
        self.preprocessing_info['outliers'] = {
            'method': method,
            'details': outliers_info
        }
        
        return self.processed_data
    
    def normalize_data(self, method='standard', columns=None):
        """
        数据标准化/归一化
        
        Args:
            method (str): 标准化方法 ('standard', 'minmax', 'robust')
            columns (list): 要标准化的列名
        """
        if columns is None:
            columns = self.processed_data.columns
        
        print(f"使用 {method} 方法进行数据标准化...")
        
        # 保存原始数据用于对比
        original_stats = self.processed_data[columns].describe()
        
        if method == 'standard':
            # 标准化 (均值0，标准差1)
            scaler = StandardScaler()
            
        elif method == 'minmax':
            # 最小-最大归一化 (0-1)
            scaler = MinMaxScaler()
            
        elif method == 'robust':
            # 鲁棒标准化 (中位数和四分位距)
            scaler = RobustScaler()
        
        # 拟合和转换数据
        scaled_data = scaler.fit_transform(self.processed_data[columns])
        
        # 更新数据
        self.processed_data[columns] = scaled_data
        
        # 保存标准化器
        self.scalers[method] = scaler
        
        # 记录标准化信息
        self.preprocessing_info['normalization'] = {
            'method': method,
            'columns': columns,
            'original_stats': original_stats,
            'scaled_stats': self.processed_data[columns].describe()
        }
        
        print(f"数据标准化完成，影响列: {columns}")
        
        return self.processed_data
    
    def create_features(self):
        """创建新特征"""
        print("创建新特征...")
        
        # 时间特征
        self.processed_data['hour'] = self.processed_data.index.hour
        self.processed_data['day_of_week'] = self.processed_data.index.dayofweek
        self.processed_data['month'] = self.processed_data.index.month
        self.processed_data['day_of_year'] = self.processed_data.index.dayofyear
        
        # 负载相关特征
        self.processed_data['total_useful_load'] = (
            self.processed_data['HUFL'] + 
            self.processed_data['MUFL'] + 
            self.processed_data['LUFL']
        )
        
        self.processed_data['total_useless_load'] = (
            self.processed_data['HULL'] + 
            self.processed_data['MULL'] + 
            self.processed_data['LULL']
        )
        
        self.processed_data['total_load'] = (
            self.processed_data['total_useful_load'] + 
            self.processed_data['total_useless_load']
        )
        
        self.processed_data['load_efficiency'] = (
            self.processed_data['total_useful_load'] / 
            (self.processed_data['total_load'] + 1e-8)  # 避免除零
        )
        
        # 移动平均特征
        for window in [3, 6, 12, 24]:
            self.processed_data[f'OT_ma_{window}'] = self.processed_data['OT'].rolling(
                window=window, min_periods=1
            ).mean()
            
            self.processed_data[f'total_load_ma_{window}'] = self.processed_data['total_load'].rolling(
                window=window, min_periods=1
            ).mean()
        
        # 滞后特征
        for lag in [1, 2, 3, 6, 12, 24]:
            self.processed_data[f'OT_lag_{lag}'] = self.processed_data['OT'].shift(lag)
            self.processed_data[f'total_load_lag_{lag}'] = self.processed_data['total_load'].shift(lag)
        
        print(f"创建了 {len(self.processed_data.columns) - len(self.original_data.columns)} 个新特征")
        
        return self.processed_data
    
    def plot_preprocessing_comparison(self, columns=None, save_path='results'):
        """绘制预处理前后对比图"""
        if columns is None:
            columns = ['OT', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
        
        n_cols = len(columns)
        fig, axes = plt.subplots(2, n_cols, figsize=(4*n_cols, 8))
        
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, col in enumerate(columns):
            # 原始数据分布
            axes[0, i].hist(self.original_data[col], bins=50, alpha=0.7, 
                           color='blue', label='原始数据', density=True)
            axes[0, i].set_title(f'{col} - 原始数据', fontsize=12, fontweight='bold')
            axes[0, i].set_ylabel('密度', fontsize=10)
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            
            # 处理后数据分布
            axes[1, i].hist(self.processed_data[col], bins=50, alpha=0.7, 
                           color='red', label='处理后数据', density=True)
            axes[1, i].set_title(f'{col} - 处理后数据', fontsize=12, fontweight='bold')
            axes[1, i].set_xlabel(col, fontsize=10)
            axes[1, i].set_ylabel('密度', fontsize=10)
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/preprocessing_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_preprocessing_summary(self):
        """获取预处理摘要"""
        summary = {
            'original_shape': self.original_data.shape,
            'processed_shape': self.processed_data.shape,
            'preprocessing_steps': self.preprocessing_info,
            'new_features': len(self.processed_data.columns) - len(self.original_data.columns)
        }
        
        print("=" * 50)
        print("数据预处理摘要")
        print("=" * 50)
        print(f"原始数据形状: {summary['original_shape']}")
        print(f"处理后数据形状: {summary['processed_shape']}")
        print(f"新增特征数量: {summary['new_features']}")
        
        if 'missing_values' in self.preprocessing_info:
            mv_info = self.preprocessing_info['missing_values']
            print(f"缺失值处理: {mv_info['method']} ({mv_info['before']} -> {mv_info['after']})")
        
        if 'outliers' in self.preprocessing_info:
            outliers_info = self.preprocessing_info['outliers']
            print(f"异常值处理: {outliers_info['method']}")
            for col, info in outliers_info['details'].items():
                print(f"  {col}: {info['count']} 个异常值 ({info['percent']:.2f}%)")
        
        if 'normalization' in self.preprocessing_info:
            norm_info = self.preprocessing_info['normalization']
            print(f"数据标准化: {norm_info['method']}")
            print(f"  标准化列: {norm_info['columns']}")
        
        return summary
    
    def save_processed_data(self, filepath):
        """保存处理后的数据"""
        self.processed_data.to_csv(filepath)
        print(f"处理后的数据已保存到: {filepath}")

def main():
    """主函数示例"""
    print("数据预处理模块已准备就绪")
    print("使用方法:")
    print("1. 创建DataPreprocessor实例")
    print("2. 调用各种预处理方法")
    print("3. 获取处理后的数据")

if __name__ == "__main__":
    main()
