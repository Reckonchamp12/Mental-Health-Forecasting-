# Load the dataset
ds = load_dataset("fridriik/mental-health-arg-post-quarantine-covid19-dataset")
# Access the 'train' split as a pandas DataFrame
df = ds['train'].to_pandas()

print(" Dataset loaded via Hugging Face datasets library")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())
# Keep only numeric + boolean columns (easy for ML/DL models)
df_clean = df.select_dtypes(include=["float64", "int64", "bool"]).copy()

# Choose depression outcome as target (BDI-II at T2, measured after quarantine)
target_col = "BDI2_T2"  
if target_col not in df_clean.columns:
    print(" Target column not found, available columns are:")
    print(df_clean.columns.tolist())
else:
    # Drop rows with missing depression targets
    df_clean = df_clean.dropna(subset=[target_col])

    # Fill missing values in predictors
    df_clean = df_clean.fillna(df_clean.mean())

    print("\n Preprocessed dataset ready for modeling")
    print("Shape:", df_clean.shape)
    print(df_clean.head())
  # Check for missing values
print("Missing values per column:\n", df.isna().sum())
# Basic statistics for numeric columns
print("\nNumeric stats:\n", df.describe())
# Encode categorical features (one-hot encoding)
df_encoded = pd.get_dummies(df, columns=['PROVINCE', 'REGION'], drop_first=True)
# Target for forecasting
target_col = 'DEPRESSION'
# drop or impute any remaining NaNs
df_encoded = df_encoded.dropna(subset=[target_col])
# Check correlation to target
corr_matrix = df_encoded.corr()
plt.figure(figsize=(10,6))
sns.heatmap(corr_matrix[[target_col]], annot=True, cmap='coolwarm')
plt.title("Correlation with Depression")
plt.show()
print("Processed dataset shape:", df_encoded.shape)
# Select numeric columns only
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Compute correlations with DEPRESSION
corr_with_target = df[numeric_cols].corr()['DEPRESSION'].sort_values(ascending=False)
print(corr_with_target)
#  Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#  Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Preprocessing complete")
print("X_train shape:", X_train_scaled.shape)
print("X_test shape:", X_test_scaled.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# correlation heatmap
plt.figure(figsize=(12,6))
sns.heatmap(pd.DataFrame(X_train_scaled, columns=X.columns).corr(), cmap='coolwarm')
plt.title("Feature Correlations after Encoding & Scaling")
plt.show()
