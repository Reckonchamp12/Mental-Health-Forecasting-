# a) RMSE comparison
plt.figure(figsize=(12,6))
sns.barplot(x="RMSE", y="Model", data=all_results_df, palette="viridis")
plt.title("All Models Comparison — RMSE")
plt.xlabel("RMSE")
plt.ylabel("Model")
plt.show()

# b) MAE comparison
plt.figure(figsize=(12,6))
sns.barplot(x="MAE", y="Model", data=all_results_df, palette="magma")
plt.title("All Models Comparison — MAE")
plt.xlabel("MAE")
plt.ylabel("Model")
plt.show()

# c) Combined Score comparison
plt.figure(figsize=(12,6))
sns.barplot(x="Score", y="Model", data=all_results_df, palette="coolwarm")
plt.title("All Models Comparison — Combined Score")
plt.xlabel("Score (higher=better)")
plt.ylabel("Model")
plt.show()
