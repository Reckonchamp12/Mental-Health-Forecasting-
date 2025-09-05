dl_models = {}

# 1. Simple Feedforward
dl_models['FeedForward'] = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

# 2. LSTM Simple
dl_models['LSTM'] = Sequential([
    Input(shape=(1, X_train_scaled.shape[1])),
    LSTM(50),
    Dense(1)
])

# 3. GRU Simple
dl_models['GRU'] = Sequential([
    Input(shape=(1, X_train_scaled.shape[1])),
    GRU(50),
    Dense(1)
])

# 4. SimpleRNN
dl_models['SimpleRNN'] = Sequential([
    Input(shape=(1, X_train_scaled.shape[1])),
    SimpleRNN(50),
    Dense(1)
])

# 5. LSTM with Dropout
dl_models['LSTM_Dropout'] = Sequential([
    Input(shape=(1, X_train_scaled.shape[1])),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

# 6. GRU with Dropout
dl_models['GRU_Dropout'] = Sequential([
    Input(shape=(1, X_train_scaled.shape[1])),
    GRU(50),
    Dropout(0.2),
    Dense(1)
])

# 7. LSTM stacked
dl_models['LSTM_Stacked'] = Sequential([
    Input(shape=(1, X_train_scaled.shape[1])),
    LSTM(50, return_sequences=True),
    LSTM(25),
    Dense(1)
])

# 8. GRU stacked
dl_models['GRU_Stacked'] = Sequential([
    Input(shape=(1, X_train_scaled.shape[1])),
    GRU(50, return_sequences=True),
    GRU(25),
    Dense(1)
])

# 9. FeedForward Large
dl_models['FeedForward_Large'] = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

# 10. FeedForward with Dropout
dl_models['FeedForward_Dropout'] = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1)
])

# 11. LSTM + FeedForward Hybrid
dl_models['LSTM_FF_Hybrid'] = Sequential([
    Input(shape=(1, X_train_scaled.shape[1])),
    LSTM(50),
    Dense(32, activation='relu'),
    Dense(1)
])

# 12. GRU + FeedForward Hybrid
dl_models['GRU_FF_Hybrid'] = Sequential([
    Input(shape=(1, X_train_scaled.shape[1])),
    GRU(50),
    Dense(32, activation='relu'),
    Dense(1)
])

# Store results in a list
dl_results_list = []

# Training & evaluation
epochs = 50
batch_size = 16

for name, model in dl_models.items():
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Use 3D input for RNN/LSTM/GRU, 2D for FeedForward
    train_input = X_train_dl if 'RNN' in name or 'LSTM' in name or 'GRU' in name else X_train_scaled
    test_input = X_test_dl if 'RNN' in name or 'LSTM' in name or 'GRU' in name else X_test_scaled
    
    model.fit(train_input, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0)
    
    y_pred = model.predict(test_input).flatten()
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    dl_results_list.append([name, rmse, mae, r2])
    print(f" {name} trained and evaluated")

# Convert list to DataFrame
dl_results_df = pd.DataFrame(dl_results_list, columns=['Model', 'RMSE', 'MAE', 'R2'])
dl_results_df = dl_results_df.sort_values(by='RMSE').reset_index(drop=True)

print("\nDL Models Comparison:")
print(dl_results_df)
