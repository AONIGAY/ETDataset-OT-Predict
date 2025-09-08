"""
ETT数据集时间序列预测模型训练框架
第二阶段：模型选择与训练

基于第一阶段EDA分析结果：
- OT具有明显的日周期性（24小时周期）
- 存在自相关性（ACF/PACF分析显示AR(1)特征）
- HUFL/HULL/MUFL/MULL与OT正相关，LUFL/LULL与OT负相关
- 数据分布近似正态，适合多种建模方法
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ETTDataset(Dataset):
    """时间序列数据集类"""
    def __init__(self, data, seq_len=24, pred_len=1, target_col='OT'):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_col = target_col
        self.feature_cols = [col for col in data.columns if col != target_col and col != 'date']
        
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx):
        # 输入序列
        x_features = self.data[self.feature_cols].iloc[idx:idx+self.seq_len].values
        x_target = self.data[self.target_col].iloc[idx:idx+self.seq_len].values
        
        # 目标序列
        y = self.data[self.target_col].iloc[idx+self.seq_len:idx+self.seq_len+self.pred_len].values
        
        return torch.FloatTensor(x_features), torch.FloatTensor(x_target), torch.FloatTensor(y)

class LSTMModel(nn.Module):
    """LSTM预测模型"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # 取最后一个时间步
        out = self.fc(out)
        return out

class GRUModel(nn.Module):
    """GRU预测模型"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=1, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                         batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        
        out, _ = self.gru(x, h0)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class TransformerModel(nn.Module):
    """简化版Transformer模型"""
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=3, output_size=1, dropout=0.2):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer(x)
        x = self.dropout(x[:, -1, :])  # 取最后一个时间步
        x = self.fc(x)
        return x

class ModelTrainer:
    """模型训练器"""
    def __init__(self, data_path='ETTh1.csv'):
        self.data_path = data_path
        self.data = None
        self.scaler_features = StandardScaler()
        self.scaler_target = StandardScaler()
        self.results = {}
        
    def load_and_preprocess_data(self):
        """加载和预处理数据"""
        print("加载数据...")
        self.data = pd.read_csv(self.data_path)
        self.data['date'] = pd.to_datetime(self.data['date'])
        
        # 添加时间特征（基于第一阶段发现的日周期性）
        self.data['hour'] = self.data['date'].dt.hour
        self.data['day_of_week'] = self.data['date'].dt.dayofweek
        self.data['month'] = self.data['date'].dt.month
        
        # 基于EDA结果：OT在下午15点达到峰值，凌晨5-6点最低
        self.data['hour_sin'] = np.sin(2 * np.pi * self.data['hour'] / 24)
        self.data['hour_cos'] = np.cos(2 * np.pi * self.data['hour'] / 24)
        
        print(f"数据形状: {self.data.shape}")
        print(f"时间范围: {self.data['date'].min()} 到 {self.data['date'].max()}")
        
    def create_features_for_ml(self, seq_len=24):
        """为机器学习模型创建特征"""
        feature_cols = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 
                       'hour_sin', 'hour_cos', 'day_of_week', 'month']
        
        X, y = [], []
        
        for i in range(seq_len, len(self.data)):
            # 基于ACF/PACF分析，使用24小时历史数据
            features = []
            
            # 历史特征
            for col in feature_cols:
                features.extend(self.data[col].iloc[i-seq_len:i].values)
            
            # 滞后特征（基于自相关性分析）
            features.append(self.data['OT'].iloc[i-1])  # 滞后1期
            features.append(self.data['OT'].iloc[i-24])  # 滞后24期（日周期）
            
            # 统计特征
            recent_window = self.data['OT'].iloc[i-seq_len:i]
            features.extend([
                recent_window.mean(),
                recent_window.std(),
                recent_window.max(),
                recent_window.min()
            ])
            
            X.append(features)
            y.append(self.data['OT'].iloc[i])
            
        return np.array(X), np.array(y)
    
    def split_data(self, test_ratio=0.2, val_ratio=0.2):
        """时间序列数据分割"""
        n = len(self.data)
        train_end = int(n * (1 - test_ratio - val_ratio))
        val_end = int(n * (1 - test_ratio))
        
        self.train_data = self.data.iloc[:train_end]
        self.val_data = self.data.iloc[train_end:val_end]
        self.test_data = self.data.iloc[val_end:]
        
        print(f"训练集: {len(self.train_data)} 样本")
        print(f"验证集: {len(self.val_data)} 样本")
        print(f"测试集: {len(self.test_data)} 样本")
        
    def train_ml_models(self):
        """训练传统机器学习模型"""
        print("\n=== 训练传统机器学习模型 ===")
        
        # 准备特征
        X, y = self.create_features_for_ml()
        
        # 分割数据
        n_train = len(self.train_data) - 24
        n_val = len(self.val_data)
        
        X_train = X[:n_train]
        y_train = y[:n_train]
        X_val = X[n_train:n_train+n_val]
        y_val = y[n_train:n_train+n_val]
        X_test = X[n_train+n_val:]
        y_test = y[n_train+n_val:]
        
        # 特征缩放
        X_train = self.scaler_features.fit_transform(X_train)
        X_val = self.scaler_features.transform(X_val)
        X_test = self.scaler_features.transform(X_test)
        
        # 模型字典
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf', C=1.0)
        }
        
        # 训练和评估
        for name, model in models.items():
            print(f"\n训练 {name}...")
            
            # 超参数调优（针对重点模型）
            if name == 'Random Forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
                model = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
            elif name == 'SVR':
                param_grid = {
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto'],
                    'epsilon': [0.01, 0.1, 0.2]
                }
                model = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
            
            model.fit(X_train, y_train)
            
            # 预测
            y_pred_val = model.predict(X_val)
            y_pred_test = model.predict(X_test)
            
            # 评估
            val_mse = mean_squared_error(y_val, y_pred_val)
            test_mse = mean_squared_error(y_test, y_pred_test)
            val_mae = mean_absolute_error(y_val, y_pred_val)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            val_mape = mean_absolute_percentage_error(y_val, y_pred_val)
            test_mape = mean_absolute_percentage_error(y_test, y_pred_test)
            
            self.results[name] = {
                'val_mse': val_mse,
                'test_mse': test_mse,
                'val_mae': val_mae,
                'test_mae': test_mae,
                'val_mape': val_mape,
                'test_mape': test_mape,
                'val_rmse': np.sqrt(val_mse),
                'test_rmse': np.sqrt(test_mse)
            }
            
            print(f"验证集 - MSE: {val_mse:.4f}, RMSE: {np.sqrt(val_mse):.4f}, MAE: {val_mae:.4f}, MAPE: {val_mape:.4f}")
            print(f"测试集 - MSE: {test_mse:.4f}, RMSE: {np.sqrt(test_mse):.4f}, MAE: {test_mae:.4f}, MAPE: {test_mape:.4f}")
    
    def train_deep_learning_models(self, seq_len=24, batch_size=64, epochs=50):
        """训练深度学习模型"""
        print("\n=== 训练深度学习模型 ===")
        
        # 数据缩放
        feature_cols = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
        
        combined_data = pd.concat([self.train_data, self.val_data, self.test_data])
        combined_features = combined_data[feature_cols + ['OT']]
        
        # 分别缩放特征和目标
        scaled_features = self.scaler_features.fit_transform(combined_features[feature_cols])
        scaled_target = self.scaler_target.fit_transform(combined_features[['OT']])
        
        combined_scaled = np.hstack([scaled_features, scaled_target])
        combined_scaled_df = pd.DataFrame(
            combined_scaled, 
            columns=feature_cols + ['OT'],
            index=combined_data.index
        )
        
        train_scaled = combined_scaled_df.loc[self.train_data.index]
        val_scaled = combined_scaled_df.loc[self.val_data.index]
        test_scaled = combined_scaled_df.loc[self.test_data.index]
        
        # 创建数据集
        train_dataset = ETTDataset(train_scaled, seq_len)
        val_dataset = ETTDataset(val_scaled, seq_len)
        test_dataset = ETTDataset(test_scaled, seq_len)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        input_size = len(feature_cols)
        
        # 深度学习模型
        dl_models = {
            'LSTM': LSTMModel(input_size, hidden_size=128, num_layers=2),
            'GRU': GRUModel(input_size, hidden_size=128, num_layers=2),
            'Transformer': TransformerModel(input_size, d_model=128, nhead=8, num_layers=3)
        }
        
        for name, model in dl_models.items():
            print(f"\n训练 {name}...")
            
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
            
            # 训练循环
            train_losses = []
            val_losses = []
            
            for epoch in range(epochs):
                # 训练阶段
                model.train()
                train_loss = 0
                for x_features, x_target, y in train_loader:
                    optimizer.zero_grad()
                    
                    # 将特征和历史目标拼接
                    x_combined = torch.cat([x_features, x_target.unsqueeze(-1)], dim=-1)
                    
                    outputs = model(x_combined)
                    loss = criterion(outputs.squeeze(), y.squeeze())
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                # 验证阶段
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for x_features, x_target, y in val_loader:
                        x_combined = torch.cat([x_features, x_target.unsqueeze(-1)], dim=-1)
                        outputs = model(x_combined)
                        loss = criterion(outputs.squeeze(), y.squeeze())
                        val_loss += loss.item()
                
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                
                scheduler.step(val_loss)
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # 测试评估
            model.eval()
            predictions = []
            actuals = []
            
            with torch.no_grad():
                for x_features, x_target, y in test_loader:
                    x_combined = torch.cat([x_features, x_target.unsqueeze(-1)], dim=-1)
                    outputs = model(x_combined)
                    predictions.extend(outputs.squeeze().numpy())
                    actuals.extend(y.squeeze().numpy())
            
            # 反标准化
            predictions = np.array(predictions).reshape(-1, 1)
            actuals = np.array(actuals).reshape(-1, 1)
            
            predictions = self.scaler_target.inverse_transform(predictions).flatten()
            actuals = self.scaler_target.inverse_transform(actuals).flatten()
            
            # 计算指标
            test_mse = mean_squared_error(actuals, predictions)
            test_mae = mean_absolute_error(actuals, predictions)
            test_mape = mean_absolute_percentage_error(actuals, predictions)
            test_rmse = np.sqrt(test_mse)
            
            self.results[name] = {
                'test_mse': test_mse,
                'test_mae': test_mae,
                'test_mape': test_mape,
                'test_rmse': test_rmse,
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            
            print(f"测试集 - MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, MAPE: {test_mape:.4f}")
    
    def compare_models(self):
        """模型性能对比"""
        print("\n=== 模型性能对比 ===")
        
        comparison_df = pd.DataFrame(self.results).T
        
        # 选择主要指标进行对比
        main_metrics = ['test_mse', 'test_rmse', 'test_mae', 'test_mape']
        if all(col in comparison_df.columns for col in main_metrics):
            comparison_df = comparison_df[main_metrics]
        
        print("\n各模型测试集性能:")
        print(comparison_df.round(4))
        
        # 找出最佳模型
        if 'test_rmse' in comparison_df.columns:
            best_model = comparison_df['test_rmse'].idxmin()
            print(f"\n最佳模型 (基于RMSE): {best_model}")
            print(f"RMSE: {comparison_df.loc[best_model, 'test_rmse']:.4f}")
        
        # 可视化对比
        if len(comparison_df) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('模型性能对比', fontsize=16)
            
            for i, metric in enumerate(main_metrics):
                if metric in comparison_df.columns:
                    ax = axes[i//2, i%2]
                    comparison_df[metric].plot(kind='bar', ax=ax)
                    ax.set_title(f'{metric.upper()} 对比')
                    ax.set_ylabel(metric.upper())
                    ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.show()
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        dl_models = ['LSTM', 'GRU', 'Transformer']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for i, model_name in enumerate(dl_models):
            if model_name in self.results and 'train_losses' in self.results[model_name]:
                ax = axes[i]
                train_losses = self.results[model_name]['train_losses']
                val_losses = self.results[model_name]['val_losses']
                
                ax.plot(train_losses, label='训练损失')
                ax.plot(val_losses, label='验证损失')
                ax.set_title(f'{model_name} 训练曲线')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.legend()
                ax.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def run_full_pipeline(self):
        """运行完整的模型训练流程"""
        print("开始ETT时间序列预测模型训练...")
        
        # 1. 数据加载和预处理
        self.load_and_preprocess_data()
        
        # 2. 数据分割
        self.split_data()
        
        # 3. 训练传统机器学习模型
        self.train_ml_models()
        
        # 4. 训练深度学习模型
        self.train_deep_learning_models()
        
        # 5. 模型对比
        self.compare_models()
        
        # 6. 可视化训练过程
        self.plot_training_curves()
        
        print("\n模型训练完成！")
        return self.results

# 使用示例
if __name__ == "__main__":
    # 初始化训练器
    trainer = ModelTrainer('ETTh1.csv')  # 请根据实际路径修改
    
    # 运行完整流程
    results = trainer.run_full_pipeline()
    
    # 保存结果
    import json
    with open('model_results.json', 'w') as f:
        # 过滤掉无法序列化的内容
        serializable_results = {}
        for model, metrics in results.items():
            serializable_results[model] = {
                k: v for k, v in metrics.items() 
                if not isinstance(v, (list, np.ndarray)) or k in ['train_losses', 'val_losses']
            }
        json.dump(serializable_results, f, indent=2)
    
    print("结果已保存到 model_results.json")