#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
电力变压器状态预测系统 - 探索性数据分析可视化模块
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

class EDAVisualizer:
    """探索性数据分析可视化器"""
    
    def __init__(self, data, save_path='results'):
        """
        初始化可视化器
        
        Args:
            data (pd.DataFrame): 数据
            save_path (str): 图片保存路径
        """
        self.data = data
        self.save_path = save_path
        
        # 创建保存目录
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
    def plot_time_series(self, columns=None, sample_size=1000):
        """
        绘制时间序列图
        
        Args:
            columns (list): 要绘制的列名
            sample_size (int): 采样大小（用于大数据集）
        """
        if columns is None:
            columns = self.data.columns
        
        # 数据采样（如果数据太大）
        if len(self.data) > sample_size:
            data_sample = self.data.sample(n=sample_size).sort_index()
        else:
            data_sample = self.data
        
        # 创建子图
        n_cols = len(columns)
        fig, axes = plt.subplots(n_cols, 1, figsize=(15, 3*n_cols))
        
        if n_cols == 1:
            axes = [axes]
        
        for i, col in enumerate(columns):
            axes[i].plot(data_sample.index, data_sample[col], linewidth=1)
            axes[i].set_title(f'{col} 时间序列图', fontsize=14, fontweight='bold')
            axes[i].set_ylabel(col, fontsize=12)
            axes[i].grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_val = data_sample[col].mean()
            std_val = data_sample[col].std()
            axes[i].axhline(y=mean_val, color='red', linestyle='--', alpha=0.7, 
                          label=f'均值: {mean_val:.2f}')
            axes[i].axhline(y=mean_val + std_val, color='orange', linestyle='--', alpha=0.7,
                          label=f'+1σ: {mean_val + std_val:.2f}')
            axes[i].axhline(y=mean_val - std_val, color='orange', linestyle='--', alpha=0.7,
                          label=f'-1σ: {mean_val - std_val:.2f}')
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'time_series_plot.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_distribution(self, columns=None):
        """
        绘制分布图（直方图 + 密度图）
        
        Args:
            columns (list): 要绘制的列名
        """
        if columns is None:
            columns = self.data.columns
        
        n_cols = len(columns)
        fig, axes = plt.subplots(2, n_cols, figsize=(4*n_cols, 8))
        
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, col in enumerate(columns):
            # 直方图
            axes[0, i].hist(self.data[col], bins=50, alpha=0.7, color='skyblue', 
                           edgecolor='black', density=True)
            axes[0, i].set_title(f'{col} 直方图', fontsize=12, fontweight='bold')
            axes[0, i].set_xlabel(col, fontsize=10)
            axes[0, i].set_ylabel('密度', fontsize=10)
            axes[0, i].grid(True, alpha=0.3)
            
            # 密度图
            self.data[col].plot(kind='density', ax=axes[1, i], color='red', linewidth=2)
            axes[1, i].set_title(f'{col} 密度图', fontsize=12, fontweight='bold')
            axes[1, i].set_xlabel(col, fontsize=10)
            axes[1, i].set_ylabel('密度', fontsize=10)
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'distribution_plot.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_matrix(self):
        """绘制相关性矩阵热力图"""
        # 计算相关性矩阵
        corr_matrix = self.data.corr()
        
        # 创建热力图
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8},
                   fmt='.3f', annot_kws={'size': 10})
        
        plt.title('特征相关性矩阵', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'correlation_matrix.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return corr_matrix
    
    def plot_box_plots(self, columns=None):
        """
        绘制箱线图
        
        Args:
            columns (list): 要绘制的列名
        """
        if columns is None:
            columns = self.data.columns
        
        n_cols = len(columns)
        fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 6))
        
        if n_cols == 1:
            axes = [axes]
        
        for i, col in enumerate(columns):
            box_plot = axes[i].boxplot(self.data[col], patch_artist=True)
            box_plot['boxes'][0].set_facecolor('lightblue')
            axes[i].set_title(f'{col} 箱线图', fontsize=12, fontweight='bold')
            axes[i].set_ylabel(col, fontsize=10)
            axes[i].grid(True, alpha=0.3)
            
            # 添加统计信息
            stats = self.data[col].describe()
            axes[i].text(0.02, 0.98, f'均值: {stats["mean"]:.2f}\n'
                                    f'中位数: {stats["50%"]:.2f}\n'
                                    f'标准差: {stats["std"]:.2f}',
                        transform=axes[i].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'box_plots.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_ot_analysis(self):
        """专门分析油温(OT)特征"""
        ot_data = self.data['OT']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 油温时间序列
        axes[0, 0].plot(self.data.index, ot_data, linewidth=1, color='blue')
        axes[0, 0].set_title('油温时间序列', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('油温 (°C)', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 添加趋势线
        z = np.polyfit(range(len(ot_data)), ot_data, 1)
        p = np.poly1d(z)
        axes[0, 0].plot(self.data.index, p(range(len(ot_data))), 
                       "r--", alpha=0.8, linewidth=2, label='趋势线')
        axes[0, 0].legend()
        
        # 2. 油温分布
        axes[0, 1].hist(ot_data, bins=50, alpha=0.7, color='green', 
                       edgecolor='black', density=True)
        axes[0, 1].set_title('油温分布', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('油温 (°C)', fontsize=12)
        axes[0, 1].set_ylabel('密度', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 油温与负载的关系
        axes[1, 0].scatter(self.data['HUFL'], ot_data, alpha=0.5, s=1)
        axes[1, 0].set_title('油温 vs 高有用负载', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('高有用负载', fontsize=12)
        axes[1, 0].set_ylabel('油温 (°C)', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 油温统计信息
        stats_text = f"""
        统计信息:
        均值: {ot_data.mean():.2f}°C
        中位数: {ot_data.median():.2f}°C
        标准差: {ot_data.std():.2f}°C
        最小值: {ot_data.min():.2f}°C
        最大值: {ot_data.max():.2f}°C
        偏度: {ot_data.skew():.3f}
        峰度: {ot_data.kurtosis():.3f}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 1].set_title('油温统计信息', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'ot_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_seasonal_analysis(self):
        """季节性分析"""
        # 提取时间特征
        data_with_time = self.data.copy()
        data_with_time['hour'] = data_with_time.index.hour
        data_with_time['day'] = data_with_time.index.dayofweek
        data_with_time['month'] = data_with_time.index.month
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 按小时分析
        hourly_ot = data_with_time.groupby('hour')['OT'].mean()
        axes[0, 0].plot(hourly_ot.index, hourly_ot.values, marker='o', linewidth=2)
        axes[0, 0].set_title('油温按小时变化', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('小时', fontsize=12)
        axes[0, 0].set_ylabel('平均油温 (°C)', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 按星期分析
        daily_ot = data_with_time.groupby('day')['OT'].mean()
        day_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        axes[0, 1].bar(range(7), daily_ot.values, color='skyblue', edgecolor='black')
        axes[0, 1].set_title('油温按星期变化', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('星期', fontsize=12)
        axes[0, 1].set_ylabel('平均油温 (°C)', fontsize=12)
        axes[0, 1].set_xticks(range(7))
        axes[0, 1].set_xticklabels(day_names)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 按月份分析
        monthly_ot = data_with_time.groupby('month')['OT'].mean()
        axes[1, 0].plot(monthly_ot.index, monthly_ot.values, marker='s', 
                       linewidth=2, markersize=8, color='red')
        axes[1, 0].set_title('油温按月份变化', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('月份', fontsize=12)
        axes[1, 0].set_ylabel('平均油温 (°C)', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 负载与油温关系
        axes[1, 1].scatter(data_with_time['HUFL'], data_with_time['OT'], 
                          alpha=0.3, s=1, color='purple')
        axes[1, 1].set_title('负载与油温关系', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('高有用负载', fontsize=12)
        axes[1, 1].set_ylabel('油温 (°C)', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'seasonal_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        print("=" * 60)
        print("探索性数据分析报告")
        print("=" * 60)
        
        # 1. 时间序列图
        print("1. 生成时间序列图...")
        self.plot_time_series()
        
        # 2. 分布图
        print("2. 生成分布图...")
        self.plot_distribution()
        
        # 3. 相关性矩阵
        print("3. 生成相关性矩阵...")
        corr_matrix = self.plot_correlation_matrix()
        
        # 4. 箱线图
        print("4. 生成箱线图...")
        self.plot_box_plots()
        
        # 5. 油温专门分析
        print("5. 生成油温分析图...")
        self.plot_ot_analysis()
        
        # 6. 季节性分析
        print("6. 生成季节性分析图...")
        self.plot_seasonal_analysis()
        
        print(f"\n所有图表已保存到: {self.save_path}")
        
        return corr_matrix

def main():
    """主函数"""
    # 这里需要先运行data_analysis.py来加载数据
    print("请先运行 data_analysis.py 来加载数据，然后使用此模块进行可视化分析")

if __name__ == "__main__":
    main()
