# 电力变压器状态预测系统 - 第一阶段

## 项目概述
本项目旨在设计一个电力变压器状态预测系统，使用ETDataset数据集进行油温预测。

## 第一阶段任务
1. 项目准备：熟悉任务书要求和目标
2. 环境配置：安装必要库
3. 数据获取：检查ETDataset数据
4. 熟悉数据格式（时间序列、电力指标、油温数据等）
5. 数据预处理：检查缺失值、异常值
6. 数据清洗、标准化/归一化
7. 画图（折线图、直方图）做探索性分析（EDA）

## 数据集说明
- **ETT-small**: 包含2个电力变压器的数据，时间跨度2016年7月-2018年7月
- **数据特征**:
  - date: 时间戳
  - HUFL: 高有用负载 (High Useful Load)
  - HULL: 高无用负载 (High Useless Load)
  - MUFL: 中有用负载 (Middle Useful Load)
  - MULL: 中无用负载 (Middle Useless Load)
  - LUFL: 低有用负载 (Low Useful Load)
  - LULL: 低无用负载 (Low Useless Load)
  - OT: 油温 (Oil Temperature) - 预测目标

## 环境安装
```bash
pip install -r requirements.txt
```

## 文件结构
```
project/first/
├── requirements.txt          # 依赖包列表
├── README.md                # 项目说明
├── data_analysis.py         # 数据分析和预处理
├── eda_visualization.py     # 探索性数据分析可视化
└── results/                 # 分析结果和图表
```
