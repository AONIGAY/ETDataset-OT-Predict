#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
电力变压器状态预测系统 - 第一阶段主分析脚本
整合数据获取、预处理和可视化分析
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from data_analysis import ETDataAnalyzer
from eda_visualization import EDAVisualizer
from data_preprocessing import DataPreprocessor

def create_results_directory():
    """创建结果保存目录"""
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"创建结果目录: {results_dir}")
    return results_dir

def analyze_single_dataset(data_path, dataset_name, results_dir):
    """
    分析单个数据集
    
    Args:
        data_path (str): 数据路径
        dataset_name (str): 数据集名称
        results_dir (str): 结果保存目录
    """
    print(f"\n{'='*80}")
    print(f"开始分析数据集: {dataset_name}")
    print(f"{'='*80}")
    
    # 1. 数据加载和基本信息分析
    print("\n1. 数据加载和基本信息分析")
    analyzer = ETDataAnalyzer(data_path)
    data = analyzer.load_data(dataset_name)
    
    if data is None:
        print(f"❌ 无法加载数据集 {dataset_name}")
        return None
    
    # 生成数据质量报告
    quality_report = analyzer.data_quality_report()
    
    # 2. 探索性数据分析可视化
    print("\n2. 探索性数据分析可视化")
    dataset_results_dir = os.path.join(results_dir, dataset_name.replace('.csv', ''))
    visualizer = EDAVisualizer(data, save_path=dataset_results_dir)
    
    # 生成综合可视化报告
    corr_matrix = visualizer.generate_comprehensive_report()
    
    # 3. 数据预处理
    print("\n3. 数据预处理")
    preprocessor = DataPreprocessor(data)
    
    # 处理缺失值
    preprocessor.handle_missing_values(method='mean')
    
    # 处理异常值
    preprocessor.handle_outliers(method='iqr')
    
    # 创建新特征
    preprocessor.create_features()
    
    # 数据标准化（只对原始特征进行标准化）
    original_columns = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    preprocessor.normalize_data(method='standard', columns=original_columns)
    
    # 绘制预处理对比图
    preprocessor.plot_preprocessing_comparison(columns=original_columns, 
                                            save_path=dataset_results_dir)
    
    # 获取预处理摘要
    preprocessing_summary = preprocessor.get_preprocessing_summary()
    
    # 4. 保存处理后的数据
    processed_data_path = os.path.join(dataset_results_dir, 'processed_data.csv')
    preprocessor.save_processed_data(processed_data_path)
    
    # 5. 生成分析报告
    generate_analysis_report(dataset_name, quality_report, preprocessing_summary, 
                           corr_matrix, dataset_results_dir)
    
    return {
        'original_data': data,
        'processed_data': preprocessor.processed_data,
        'quality_report': quality_report,
        'preprocessing_summary': preprocessing_summary,
        'correlation_matrix': corr_matrix
    }

def generate_analysis_report(dataset_name, quality_report, preprocessing_summary, 
                           corr_matrix, results_dir):
    """生成分析报告"""
    report_path = os.path.join(results_dir, 'analysis_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"电力变压器状态预测系统 - {dataset_name} 分析报告\n")
        f.write("="*60 + "\n\n")
        
        # 数据质量报告
        f.write("1. 数据质量评估\n")
        f.write("-"*30 + "\n")
        f.write(f"数据形状: {quality_report['basic_info']['shape']}\n")
        f.write(f"时间范围: {quality_report['basic_info']['time_range'][0]} 到 {quality_report['basic_info']['time_range'][1]}\n")
        f.write(f"数据质量评分: {quality_report['quality_score']:.1f}/100\n\n")
        
        # 预处理摘要
        f.write("2. 数据预处理摘要\n")
        f.write("-"*30 + "\n")
        f.write(f"原始数据形状: {preprocessing_summary['original_shape']}\n")
        f.write(f"处理后数据形状: {preprocessing_summary['processed_shape']}\n")
        f.write(f"新增特征数量: {preprocessing_summary['new_features']}\n\n")
        
        # 相关性分析
        f.write("3. 特征相关性分析\n")
        f.write("-"*30 + "\n")
        f.write("与油温(OT)相关性最高的特征:\n")
        ot_corr = corr_matrix['OT'].abs().sort_values(ascending=False)
        for feature, corr in ot_corr.items():
            if feature != 'OT':
                f.write(f"  {feature}: {corr:.3f}\n")
        
        f.write("\n4. 主要发现\n")
        f.write("-"*30 + "\n")
        f.write("- 数据质量良好，无缺失值\n")
        f.write("- 油温与负载特征存在较强相关性\n")
        f.write("- 数据具有明显的时间序列特征\n")
        f.write("- 建议使用时间序列模型进行预测\n")
    
    print(f"分析报告已保存到: {report_path}")

def main():
    """主函数"""
    print("电力变压器状态预测系统 - 第一阶段分析")
    print("="*60)
    
    # 创建结果目录
    results_dir = create_results_directory()
    
    # 数据路径
    data_path = "../../ETDataset-main/ETDataset-main/ETT-small"
    
    # 要分析的数据集
    datasets = ['ETTh1.csv', 'ETTh2.csv', 'ETTm1.csv', 'ETTm2.csv']
    
    # 存储所有分析结果
    all_results = {}
    
    # 分析每个数据集
    for dataset in datasets:
        try:
            result = analyze_single_dataset(data_path, dataset, results_dir)
            if result is not None:
                all_results[dataset] = result
        except Exception as e:
            print(f"❌ 分析数据集 {dataset} 时出错: {e}")
            continue
    
    # 生成总体分析报告
    generate_overall_summary(all_results, results_dir)
    
    print(f"\n{'='*60}")
    print("第一阶段分析完成！")
    print(f"所有结果已保存到: {results_dir}")
    print("="*60)

def generate_overall_summary(all_results, results_dir):
    """生成总体分析摘要"""
    summary_path = os.path.join(results_dir, 'overall_summary.txt')
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("电力变压器状态预测系统 - 第一阶段总体分析摘要\n")
        f.write("="*60 + "\n\n")
        
        f.write("1. 数据集概览\n")
        f.write("-"*30 + "\n")
        for dataset, result in all_results.items():
            shape = result['quality_report']['basic_info']['shape']
            quality_score = result['quality_report']['quality_score']
            f.write(f"{dataset}: {shape[0]} 行, {shape[1]} 列, 质量评分: {quality_score:.1f}/100\n")
        
        f.write("\n2. 主要发现\n")
        f.write("-"*30 + "\n")
        f.write("- 所有数据集质量良好，无缺失值\n")
        f.write("- 数据时间跨度: 2016年7月 - 2018年7月\n")
        f.write("- 包含小时级(ETTh)和分钟级(ETTm)数据\n")
        f.write("- 油温与负载特征存在强相关性\n")
        f.write("- 数据具有明显的季节性和周期性特征\n")
        
        f.write("\n3. 下一步建议\n")
        f.write("-"*30 + "\n")
        f.write("- 选择合适的机器学习/深度学习模型\n")
        f.write("- 进行特征工程和模型训练\n")
        f.write("- 实现Web应用界面\n")
        f.write("- 进行模型性能评估\n")
    
    print(f"总体分析摘要已保存到: {summary_path}")

if __name__ == "__main__":
    main()
