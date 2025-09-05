# List of models
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
    "SVR": SVR(kernel='rbf', C=1.0, epsilon=0.1),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "DecisionTree": DecisionTreeRegressor(random_state=42)
}

# List to store results
ml_results_list = []

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Append results to list
    ml_results_list.append([name, rmse, mae, r2])
    
    print(f" {name} done")

# Convert list to DataFrame
ml_results_df = pd.DataFrame(ml_results_list, columns=['Model', 'RMSE', 'MAE', 'R2'])
ml_results_df = ml_results_df.sort_values(by='RMSE').reset_index(drop=True)

print("\nTop ML models by RMSE:")
print(ml_results_df)
# Bar plot comparison
plt.figure(figsize=(12,6))
sns.barplot(x= "RMSE", y="Model", data=ml_results_df, palette="viridis")
plt.title("ML Model Comparison — RMSE")
plt.show()
