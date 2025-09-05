# Dictionary to hold fused models
fused_models = {}

# 1. LSTM + Dense
model1 = Sequential([LSTM(64, input_shape=(1, X_train_scaled.shape[1])),
                     Dense(32, activation='relu'),
                     Dense(1)])
fused_models['LSTM_FF_1'] = model1

# 2. LSTM stacked + Dense
model2 = Sequential([LSTM(64, return_sequences=True, input_shape=(1, X_train_scaled.shape[1])),
                     LSTM(32),
                     Dense(1)])
fused_models['LSTM_Stacked_FF'] = model2

# 3. LSTM + Dropout + Dense
model3 = Sequential([LSTM(64, input_shape=(1, X_train_scaled.shape[1])),
                     Dropout(0.2),
                     Dense(32, activation='relu'),
                     Dense(1)])
fused_models['LSTM_DO_FF'] = model3

# 4. RNN + Dense
model4 = Sequential([SimpleRNN(64, input_shape=(1, X_train_scaled.shape[1])),
                     Dense(32, activation='relu'),
                     Dense(1)])
fused_models['RNN_FF_1'] = model4

# 5. RNN stacked + Dense
model5 = Sequential([SimpleRNN(64, return_sequences=True, input_shape=(1, X_train_scaled.shape[1])),
                     SimpleRNN(32),
                     Dense(1)])
fused_models['RNN_Stacked_FF'] = model5

# 6. RNN + Dropout + Dense
model6 = Sequential([SimpleRNN(64, input_shape=(1, X_train_scaled.shape[1])),
                     Dropout(0.2),
                     Dense(32, activation='relu'),
                     Dense(1)])
fused_models['RNN_DO_FF'] = model6

# 7. LSTM + RNN hybrid
model7 = Sequential([LSTM(64, return_sequences=True, input_shape=(1, X_train_scaled.shape[1])),
                     SimpleRNN(32),
                     Dense(1)])
fused_models['LSTM_RNN_Fused'] = model7

# 8. LSTM + GRU fused
model8 = Sequential([LSTM(64, return_sequences=True, input_shape=(1, X_train_scaled.shape[1])),
                     GRU(32),
                     Dense(1)])
fused_models['LSTM_GRU_Fused'] = model8

# 9. RNN + LSTM + Dense
model9 = Sequential([SimpleRNN(64, return_sequences=True, input_shape=(1, X_train_scaled.shape[1])),
                     LSTM(32),
                     Dense(1)])
fused_models['RNN_LSTM_Fused'] = model9

# 10. LSTM + Dense + Dropout
model10 = Sequential([LSTM(64, input_shape=(1, X_train_scaled.shape[1])),
                      Dense(32, activation='relu'),
                      Dropout(0.2),
                      Dense(1)])
fused_models['LSTM_FF_DO'] = model10

# Training parameters
epochs = 50
batch_size = 16
fused_results_list = []

for name, model in fused_models.items():
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Train the model
    model.fit(X_train_dl, y_train, epochs=epochs, batch_size=batch_size,
              validation_split=0.1, verbose=0)
    
    # Predict on test data
    y_pred = model.predict(X_test_dl).flatten()
    
    # Compute metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Append results to list
    fused_results_list.append([name, rmse, mae, r2])
    print(f" {name} trained")

# Convert list to DataFrame
fused_results_df = pd.DataFrame(fused_results_list, columns=['Model', 'RMSE', 'MAE', 'R2'])
fused_results_df = fused_results_df.sort_values(by='RMSE').reset_index(drop=True)

print("\nTop Fused Models Comparison:")
print(fused_results_df)
