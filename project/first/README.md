# 第一阶段：数据预处理与探索性分析（EDA）

本阶段目标：

- 熟悉 ETT 数据格式（时间序列、电力负荷指标、油温 OT）。
- 完成缺失值/异常值检测与清洗。
- 完成特征缩放（标准化/归一化），导出清洗/缩放后的数据。
- 完成与 OT 的相关性与时序特性分析，并导出图表。

参考数据集：`ETDataset` 仓库（`https://github.com/zhouhaoyi/ETDataset`）。

## 环境

- Conda 环境：`ETD-Env`
- 主要依赖：pandas、numpy、matplotlib、seaborn、scikit-learn、scipy、statsmodels、jupyter、notebook、plotly、tqdm、openpyxl、ipykernel

若需（重新）注册 Jupyter 内核：

```bash
conda run -n ETD-Env python -m ipykernel install --user --name ETD-Env --display-name "Python (ETD-Env)"
```

## 数据

- 位置：`ETDataset-main/ETT-small/`
- 文件：`ETTh1.csv`、`ETTh2.csv`、`ETTm1.csv`、`ETTm2.csv`
- 字段说明与背景详见官方仓库文档。

## 使用步骤

1. 打开 Notebook：`project/first/notebooks/01_data_preprocess_eda.ipynb`
2. 选择内核：`Python (ETD-Env)`
3. 在第二个单元将 `CSV_FILE` 设置为需要处理的数据文件名（如 `ETTh1.csv`）。
4. 可切换异常处理与缩放策略：
   - `REMOVE_OUTLIER_ROWS`：是否删除含异常的样本行（默认 False）
   - `WINSORIZE`：是否对数值列进行截尾（默认 True）
   - `USE_STANDARD_SCALER`：True 使用 StandardScaler；False 使用 MinMaxScaler
5. 依次运行全部单元。

## 输出

- 清洗/缩放后的数据：`project/first/outputs/cleaned/`
  - `*_clean.csv`：清洗后数据
  - `*_scaled.csv`：缩放后数据
- 图表：`project/first/outputs/figures/`
  - `*_ot_overview.png`：OT 概览折线
  - `*_ot_box_hist.png`：OT 箱线图与分布
  - `*_corr_heatmap.png`：特征相关性热力图
  - `*_ot_scatter.png`：OT 与各负荷散点关系
  - `*_ot_acf_pacf.png`：OT 的 ACF/PACF
  - `*_ot_hourly_mean.png`：OT 按小时均值

## 结果要点（示例占位，运行后据实填写）

- 缺失值比例：…；异常样本占比：…
- 与 OT 相关性最高的特征：…（正/负相关）
- ADF 检验：p-value=…（平稳/非平稳）
- 显著周期性：…（例如日/周周期）

## 参考

- ETT 数据集官方仓库：`https://github.com/zhouhaoyi/ETDataset`


