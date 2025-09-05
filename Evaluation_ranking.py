# 1️ Combine all results
all_results_df = pd.concat([results_df, dl_results_df, fused_results_df], ignore_index=True)

# 2️ Normalize metrics for fair comparison
# RMSE & MAE -> lower is better, R2 -> higher is better
scaler = MinMaxScaler()
metrics_scaled = all_results_df[['RMSE', 'MAE', 'R2']].copy()

# Invert RMSE and MAE to make higher=better
metrics_scaled[['RMSE', 'MAE']] = -metrics_scaled[['RMSE', 'MAE']]
metrics_scaled[['RMSE_scaled', 'MAE_scaled', 'R2_scaled']] = scaler.fit_transform(metrics_scaled)
all_results_df[['RMSE_scaled', 'MAE_scaled', 'R2_scaled']] = metrics_scaled[['RMSE_scaled', 'MAE_scaled', 'R2_scaled']]

# 3️ Compute combined score
# Higher score = better performance
all_results_df['Score'] = all_results_df['RMSE_scaled'] + all_results_df['MAE_scaled'] + all_results_df['R2_scaled']

# Rank models by Score
all_results_df = all_results_df.sort_values(by='Score', ascending=False).reset_index(drop=True)

# 4️ Display top 10 models
print(" Top 10 Models by Forecasting Power:")
print(all_results_df[['Model', 'RMSE', 'MAE', 'R2', 'Score']].head(10))
