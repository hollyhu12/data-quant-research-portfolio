import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

current_dataset = 'Electricity'

file_mapping = {
    'Weather': os.path.join(base_dir, 'weather.csv'),
    'ETTm1': os.path.join(base_dir, 'ETTm1.csv'),
    'Electricity': os.path.join(base_dir, 'electricity.csv')
}


lookback = 24
horizons = [96, 192, 336, 720]
segment_size = 5000
plot_last_n = 300

file = file_mapping[current_dataset]
print(f'Processing {current_dataset} dataset in segments with DLinear...')

df_raw = pd.read_csv(file)

time_col = None
for col in df_raw.columns:
    if col.lower() == 'date':
        time_col = col
        break
if time_col is None:
    for col in df_raw.columns:
        if df_raw[col].dtype == 'object':
            time_col = col
            break

time_values = None
if time_col is not None:
    time_values = pd.to_datetime(df_raw[time_col], errors='coerce')

numeric_df = df_raw.select_dtypes(include=[np.number]).copy()

if 'OT' in numeric_df.columns:
    target_col = 'OT'
else:
    target_col = numeric_df.columns[-1]

data_x = numeric_df.values.astype(np.float32)
data_y = numeric_df[target_col].values.astype(np.float32)

n_samples, n_features = data_x.shape

results = []

for horizon in horizons:
    y_true_all = []
    y_pred_all = []

    last_actual = None
    last_pred = None
    last_time = None

    for start in range(0, n_samples - lookback - horizon + 1, segment_size):
        end = min(start + segment_size, n_samples - lookback - horizon + 1)

        X_list = []
        y_list = []
        idx_list = []

        for i in range(start, end):
            X_list.append(data_x[i:i+lookback, :].reshape(-1))
            y_list.append(data_y[i+lookback:i+lookback+horizon])
            idx_list.append(i)

        X_batch = np.array(X_list, dtype=np.float32)
        y_batch = np.array(y_list, dtype=np.float32)

        model = LinearRegression()
        model.fit(X_batch, y_batch)
        y_pred_batch = model.predict(X_batch)

        y_true_all.extend(y_batch.reshape(-1).tolist())
        y_pred_all.extend(y_pred_batch.reshape(-1).tolist())

        last_actual = y_batch[-1]
        last_pred = y_pred_batch[-1]

        last_start_idx = idx_list[-1] + lookback
        last_end_idx = last_start_idx + horizon

        if time_values is not None:
            last_time = time_values.iloc[last_start_idx:last_end_idx].tolist()
        else:
            last_time = list(range(last_start_idx, last_end_idx))

        del X_batch, y_batch, y_pred_batch, X_list, y_list, idx_list

    y_true_all = np.array(y_true_all, dtype=np.float32)
    y_pred_all = np.array(y_pred_all, dtype=np.float32)

    mae = mean_absolute_error(y_true_all, y_pred_all)
    mse = mean_squared_error(y_true_all, y_pred_all)
    rmse = np.sqrt(mse)
    mape = np.mean(
        np.abs((y_true_all - y_pred_all) / np.clip(np.abs(y_true_all), 1e-8, None))
    ) * 100

    results.append({
        'Dataset': current_dataset,
        'Target': target_col,
        'Horizon': horizon,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    })

    if last_actual is not None and last_pred is not None:
        if len(last_actual) > plot_last_n:
            plot_actual = last_actual[-plot_last_n:]
            plot_pred = last_pred[-plot_last_n:]
            plot_time = last_time[-plot_last_n:]
        else:
            plot_actual = last_actual
            plot_pred = last_pred
            plot_time = last_time

        plt.figure(figsize=(14, 6))
        plt.plot(plot_time, plot_actual, label='Actual', linewidth=2)
        plt.plot(plot_time, plot_pred, label='Predicted', linewidth=2)
        plt.title(
            f'{current_dataset}: Actual vs Predicted (pred_len={horizon}, last {len(plot_actual)} points)',
            fontsize=16
        )
        plt.xlabel('Time', fontsize=12)
        plt.ylabel(target_col, fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(f'{current_dataset}_pred_len_{horizon}.png', dpi=300)
        plt.show()

summary_df = pd.DataFrame(results)
summary_df.to_csv(f'{current_dataset}_DLinear_summary_metrics.csv', index=False)

print(summary_df)
print(f'Results for {current_dataset} saved to {current_dataset}_DLinear_summary_metrics.csv.')