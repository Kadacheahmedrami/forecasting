import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.linear_model import Ridge
from scipy import stats


# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def mape_loss(y_true, y_pred):
    """Custom MAPE loss function for Keras"""
    epsilon = tf.keras.backend.epsilon()  # Small constant to avoid division by zero
    diff = tf.abs((y_true - y_pred) / tf.maximum(tf.abs(y_true), epsilon))
    return 100.0 * tf.reduce_mean(diff, axis=-1)

def symmetric_mape_loss(y_true, y_pred):
    """Symmetric MAPE loss that treats over/under predictions more equally"""
    epsilon = tf.keras.backend.epsilon()
    diff = tf.abs(y_true - y_pred)
    sum_val = tf.maximum(tf.abs(y_true) + tf.abs(y_pred), epsilon)
    return 200.0 * tf.reduce_mean(diff / sum_val, axis=-1)






class StockForecaster:
    def __init__(self, seq_length=30):
        self.seq_length = seq_length
        self.scaler = RobustScaler()
        self.target_scaler = RobustScaler()
        
    def preprocess_data(self, df):
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Quarter'] = df['Date'].dt.quarter
        
        df['Month_sin'] = np.sin(2 * np.pi * df['Month']/12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month']/12)
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek']/7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek']/7)
        
        df['MA7'] = df['Target'].rolling(window=7).mean()
        df['MA21'] = df['Target'].rolling(window=21).mean()
        df['MA50'] = df['Target'].rolling(window=50).mean()
        df['RSI'] = self._calculate_rsi(df['Target'])
        df['Volatility'] = df['Target'].rolling(window=7).std()
        
        df['EMA12'] = df['Target'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Target'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        df['Price_Change'] = df['Target'].pct_change()
        df['Price_Change_3d'] = df['Target'].pct_change(periods=3)
        df['Price_Change_7d'] = df['Target'].pct_change(periods=7)
        
        for lag in [1, 2, 3, 5, 7]:
            df[f'Target_lag_{lag}'] = df['Target'].shift(lag)
        
        epsilon = 1e-8
        if 'A1' in df.columns and 'A2' in df.columns and 'A3' in df.columns and 'B1' in df.columns:
            df['A1_A2_Ratio'] = df['A1'] / (df['A2'] + epsilon)
            df['A1_A3_Ratio'] = df['A1'] / (df['A3'] + epsilon)
            df['A2_A3_Ratio'] = df['A2'] / (df['A3'] + epsilon)
            df['A_B_Ratio'] = df['A1'] / (df['B1'] + epsilon)
            df['A1_B1_Product'] = df['A1'] * df['B1']
            df['A2_B1_Product'] = df['A2'] * df['B1']
        
        df['Volatility_21d'] = df['Target'].rolling(window=21).std()
        df['Volatility_Ratio'] = df['Volatility'] / (df['Volatility_21d'] + epsilon)
        
        df['ADX'] = self._calculate_adx(df['Target'], period=14)
        
        for i in range(5):
            df[f'Day_{i}'] = (df['DayOfWeek'] == i).astype(int)
        
        if 'Volume' in df.columns:
            df['Volume_MA7'] = df['Volume'].rolling(window=7).mean()
            df['Volume_Change'] = df['Volume'].pct_change()
            df['Price_Volume_Ratio'] = df['Target'] / (df['Volume'] + epsilon)
        
        df = df.ffill().bfill()
        original_target = df['Target'].copy()
        
        if (df['Target'] <= 0).any():
            min_value = df['Target'].min()
            if min_value <= 0:
                df['Target'] = df['Target'] - min_value + 1e-3
                
        df['Target'] = np.log1p(df['Target'])
        
        features = [col for col in df.columns if col not in ['Date', 'Target']]
        
        df[features] = self.scaler.fit_transform(df[features])
        df[['Target']] = self.target_scaler.fit_transform(df[['Target']])
        
        return df, original_target
    
    def _calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gains = delta.copy()
        losses = delta.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        avg_gain = gains.rolling(window=period).mean()
        avg_loss = losses.rolling(window=period).mean()
        
        epsilon = 1e-8
        rs = avg_gain / (avg_loss + epsilon)
        
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_adx(self, price, period=14):
        high = price.rolling(window=2).max()
        low = price.rolling(window=2).min()
        
        plus_dm = high.diff()
        minus_dm = low.diff(-1)
        plus_dm = plus_dm.copy()
        minus_dm = minus_dm.copy()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': (high - price.shift(1)).abs(),
            'lc': (low - price.shift(1)).abs()
        }).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        epsilon = 1e-8
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / (atr + epsilon))
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / (atr + epsilon))
        
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + epsilon))
        adx = dx.rolling(window=period).mean()
        return adx
    
    def create_sequences(self, df):
        features = [col for col in df.columns if col not in ['Date']]
        
        X, y = [], []
        dates = []
        
        for i in range(len(df) - self.seq_length):
            X.append(df[features].iloc[i:(i + self.seq_length)].values)
            y.append(df['Target'].iloc[i + self.seq_length])
            dates.append(df['Date'].iloc[i + self.seq_length])
            
        return np.array(X), np.array(y), dates
    
    def build_model(self, input_shape):
        if isinstance(input_shape, tuple) and len(input_shape) == 2:
            seq_length, features = input_shape
            input_shape = (seq_length, features)
        else:
            input_shape = (input_shape[1], input_shape[2])
        
        inputs = tf.keras.Input(shape=input_shape)
        
        x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
        x = Dropout(0.3)(x)
        
        attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
        x = tf.keras.layers.Add()([x, attention])
        x = tf.keras.layers.LayerNormalization()(x)
        
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = Dropout(0.3)(x)
        x = Bidirectional(LSTM(32))(x)
        x = Dropout(0.3)(x)
        
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(16, activation='relu')(x)
        outputs = Dense(1)(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer='adam', 
            loss=mape_loss,  
            metrics=['mae', 'mse', tf.keras.metrics.RootMeanSquaredError()]
        )
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        model = self.build_model(X_train.shape)
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return model, history
    
    def evaluate(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        
        y_test_scaled_inv = self.target_scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_scaled_inv = self.target_scaler.inverse_transform(y_pred)
        
        y_test_inv = np.expm1(y_test_scaled_inv)
        y_pred_inv = np.expm1(y_pred_scaled_inv)
        
        mse = np.mean((y_test_inv - y_pred_inv) ** 2)
        mae = np.mean(np.abs(y_test_inv - y_pred_inv))
        rmse = np.sqrt(mse)
        
        epsilon = 1e-8
        abs_percentage_errors = np.abs((y_test_inv - y_pred_inv) / np.maximum(np.abs(y_test_inv), epsilon)) * 100
        mape = np.mean(abs_percentage_errors)
        
        direction_actual = np.sign(np.diff(np.vstack([y_test_inv[0], y_test_inv]), axis=0))
        direction_pred = np.sign(np.diff(np.vstack([y_test_inv[0], y_pred_inv]), axis=0))
        dir_compare = direction_actual == direction_pred
        dir_compare = dir_compare[direction_actual != 0]
        directional_accuracy = np.mean(dir_compare) * 100 if len(dir_compare) > 0 else 0
            
        metrics = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Directional Accuracy': directional_accuracy
        }
        
        return metrics, y_pred_inv
    
    def plot_predictions(self, dates, y_true, y_pred, metrics=None, title='Stock Price Predictions'):
        plt.figure(figsize=(15, 7))
        
        plt.plot(dates, y_true, label='Actual', marker='o', markersize=3, alpha=0.7)
        plt.plot(dates, y_pred, label='Predicted', marker='x', markersize=3, alpha=0.7)
        
        if metrics:
            metrics_str = f"MAPE: {metrics['MAPE']:.2f}%, MAE: {metrics['MAE']:.4f}, Directional Accuracy: {metrics['Directional Accuracy']:.1f}%"
            title = f"{title}\n{metrics_str}"
            
        plt.title(title, fontsize=16)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        errors = y_true.flatten() - y_pred.flatten()
        plt.figure(figsize=(12, 6))
        
        sns.histplot(errors, kde=True, bins=30)
        plt.title('Prediction Error Distribution', fontsize=16)
        plt.xlabel('Error', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        
        corr = np.corrcoef(y_true.flatten(), y_pred.flatten())[0,1]
        r2 = corr**2
        
        plt.title(f'Actual vs Predicted Values (R² = {r2:.4f})', fontsize=16)
        plt.xlabel('Actual', fontsize=12)
        plt.ylabel('Predicted', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    def analyze_feature_importance(self, X, feature_names):
        X_flat = X.reshape(X.shape[0], -1)
        
        feature_importance = {}
        
        for i, feature in enumerate(feature_names):
            feature_data = X[:, :, i]
            importance = np.mean(np.abs(feature_data))
            feature_importance[feature] = importance
            
        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        plt.figure(figsize=(12, 8))
        features = [x[0] for x in sorted_importance[:15]]
        values = [x[1] for x in sorted_importance[:15]]
        
        plt.barh(features, values)
        plt.xlabel('Mean Absolute Value')
        plt.title('Feature Importance Analysis')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        return sorted_importance


class HeterogeneousEnsemble:
    def __init__(self, seq_length=30):
        self.seq_length = seq_length
        self.lstm_forecaster = StockForecaster(seq_length)
        self.tree_models = []
        self.meta_learner = None
        
    def train(self, df):
        """Train the heterogeneous ensemble"""
        # Preprocess data using the LSTM forecaster's method
        processed_df, original_target = self.lstm_forecaster.preprocess_data(df)
        X, y, dates = self.lstm_forecaster.create_sequences(processed_df)
        
        # Check if we have enough data
        if len(X) < 10:  # Arbitrary minimal threshold
            raise ValueError(f"Not enough data for training. Only {len(X)} sequences created.")
        
        # Split data
        split_idx = int(len(X) * 0.7)
        val_idx = int(len(X) * 0.85)
        
        X_train, X_val, X_test = X[:split_idx], X[split_idx:val_idx], X[val_idx:]
        y_train, y_val, y_test = y[:split_idx], y[split_idx:val_idx], y[val_idx:]
        
        # Train LSTM model
        # Fixed: Pass X_train.shape directly
        lstm_model, _ = self.lstm_forecaster.train(X_train, y_train, X_val, y_val)
        
        # Prepare data for tree-based models (flatten 3D to 2D)
        X_tree_train = self._flatten_features(X_train)
        X_tree_val = self._flatten_features(X_val)
        X_tree_test = self._flatten_features(X_test)
        
        # Train multiple tree models with different hyperparameters
        # XGBoost models with different hyperparameters
        xgb_models = [
            xgb.XGBRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42
            ),
            xgb.XGBRegressor(
                n_estimators=100, max_depth=7, learning_rate=0.05,
                subsample=0.7, colsample_bytree=0.9, random_state=42
            ),
            xgb.XGBRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                subsample=0.9, colsample_bytree=0.7, random_state=42
            )
        ]
        
        # Train tree models
        for model in xgb_models:
            model.fit(X_tree_train, y_train)
            self.tree_models.append(model)
        
        # Generate predictions for meta-learner training
        val_preds = []
        
        # LSTM predictions
        lstm_val_preds = lstm_model.predict(X_val)
        val_preds.append(lstm_val_preds.flatten())
        
        # Tree model predictions
        for model in self.tree_models:
            tree_val_preds = model.predict(X_tree_val)
            val_preds.append(tree_val_preds)
        
        # Stack predictions for meta-learner
        stacked_preds = np.column_stack(val_preds)
        
        # Train meta-learner (weighted combination using Ridge regression)
        self.meta_learner = Ridge(alpha=1.0)
        self.meta_learner.fit(stacked_preds, y_val)
        
        # Return the trained models for further use
        return {
            'lstm_model': lstm_model,
            'tree_models': self.tree_models,
            'meta_learner': self.meta_learner
        }
    
    def _flatten_features(self, X):
        """Flatten 3D LSTM features to 2D for tree models"""
        # Create lag features from 3D tensor
        n_samples, n_timesteps, n_features = X.shape
        flattened = np.zeros((n_samples, n_timesteps * n_features))
        
        for i in range(n_samples):
            flattened[i] = X[i].flatten()
            
        return flattened
    
    def predict(self, X, lstm_model):
        """Generate ensemble predictions"""
        # Prepare tree model input
        X_tree = self._flatten_features(X)
        
        # Generate individual predictions
        all_preds = []
        
        # LSTM predictions
        lstm_preds = lstm_model.predict(X)
        all_preds.append(lstm_preds.flatten())
        
        # Tree model predictions
        for model in self.tree_models:
            tree_preds = model.predict(X_tree)
            all_preds.append(tree_preds)
        
        # Stack predictions
        stacked_preds = np.column_stack(all_preds)
        
        # Meta-learner predictions
        final_preds = self.meta_learner.predict(stacked_preds)
        
        return final_preds.reshape(-1, 1)



def calculate_metrics(y_true, y_pred):
    """Calculate performance metrics"""
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    
    # Calculate MAPE with protection against division by zero
    epsilon = 1e-8
    abs_percentage_errors = np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon)) * 100
    mape = np.mean(abs_percentage_errors)
    
    # Calculate directional accuracy
    direction_actual = np.sign(np.diff(np.vstack([y_true[0], y_true]), axis=0))
    direction_pred = np.sign(np.diff(np.vstack([y_true[0], y_pred]), axis=0))
    dir_compare = direction_actual == direction_pred
    dir_compare = dir_compare[direction_actual != 0]
    if len(dir_compare) > 0:
        directional_accuracy = np.mean(dir_compare) * 100
    else:
        directional_accuracy = 0
        
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'Directional Accuracy': directional_accuracy
    }


def main():
    print("Loading and preparing data...")
    
    try:
        # Load data
        train_df = pd.read_csv('/kaggle/input/data-bounty-1-stock-forcasting/Problem-1-train.csv')
        test_df = pd.read_csv('/kaggle/input/data-bounty-1-stock-forcasting/Problem-1-test.csv')
        
        # Display basic information about the data
        print("Training Dataset Information:")
        print(f"Total rows: {len(train_df)}")
        print(f"Date range: {train_df['Date'].min()} to {train_df['Date'].max()}")
        
        print("\nTest Dataset Information:")
        print(f"Total rows: {len(test_df)}")
        print(f"Date range: {test_df['Date'].min()} to {test_df['Date'].max()}")
        
        # Store all test dates before any processing
        test_dates_original = pd.to_datetime(test_df['Date']).tolist()
        
        # Check if needed columns exist
        required_train_cols = ['Date', 'Target']
        if not all(col in train_df.columns for col in required_train_cols):
            missing = [col for col in required_train_cols if col not in train_df.columns]
            raise ValueError(f"Required columns missing in training data: {missing}")
        
        required_test_cols = ['Date', 'A1', 'A2', 'A3', 'B1', 'B2']
        if not all(col in test_df.columns for col in required_test_cols):
            missing = [col for col in required_test_cols if col not in test_df.columns]
            raise ValueError(f"Required columns missing in test data: {missing}")
        
        print("\nTraining heterogeneous ensemble on full training data...")
        ensemble = HeterogeneousEnsemble(seq_length=30)
        trained_models = ensemble.train(train_df)
        
        print("\nPreparing test data for predictions...")
        # Create a combined dataframe by appending the last seq_length rows from train to test
        # This ensures we have enough history for the first test rows
        seq_length = ensemble.seq_length
        
        # Sort train and test dataframes by date to ensure chronological order
        train_df['Date'] = pd.to_datetime(train_df['Date'])
        test_df['Date'] = pd.to_datetime(test_df['Date'])
        train_df = train_df.sort_values('Date')
        test_df = test_df.sort_values('Date')
        
        # We need training data columns to match test data columns
        train_cols = train_df.columns.tolist()
        test_cols = test_df.columns.tolist()
        
        # Add Target column to test data using median value from training
        test_df['Target'] = train_df['Target'].median()
        
        # Add missing columns to each dataframe with NaN values
        for col in train_cols:
            if col not in test_df.columns:
                test_df[col] = np.nan
                
        for col in test_cols:
            if col not in train_df.columns:
                train_df[col] = np.nan
        
        # Get the last seq_length rows from training data
        train_tail = train_df.tail(seq_length).copy()
        
        # Combine with test data
        combined_df = pd.concat([train_tail, test_df], ignore_index=True)
        
        # Use the same preprocessing as in ensemble training
        processed_combined_df, _ = ensemble.lstm_forecaster.preprocess_data(combined_df)
        
        # Create sequences for combined data
        X_combined, _, combined_dates = ensemble.lstm_forecaster.create_sequences(processed_combined_df)
        
        print(f"Combined sequences created: {len(X_combined)}")
        
        # Generate ensemble predictions
        print("\nGenerating ensemble predictions for test data...")
        try:
            # First try with ensemble predictions
            combined_predictions = ensemble.predict(X_combined, trained_models['lstm_model'])
        except ValueError as e:
            print(f"Warning: Meta-learner prediction failed: {str(e)}")
            print("Falling back to LSTM model predictions only")
            # Use only LSTM model as fallback
            combined_predictions = trained_models['lstm_model'].predict(X_combined)
        
        # Check for and handle any remaining NaNs in predictions
        if np.isnan(combined_predictions).any():
            print("Warning: NaN values detected in predictions, replacing with median values")
            nan_mask = np.isnan(combined_predictions)
            median_val = np.nanmedian(combined_predictions)
            combined_predictions[nan_mask] = median_val if not np.isnan(median_val) else 0
        
        # Inverse transform predictions
        y_pred_scaled_inv = ensemble.lstm_forecaster.target_scaler.inverse_transform(combined_predictions)
        
        # If log transform was applied, reverse it
        y_pred_inv = np.expm1(y_pred_scaled_inv)
        
        # Match predictions back to original test dates
        date_map = {}
        for date, pred in zip(combined_dates, y_pred_inv.flatten()):
            date_map[date] = pred
            
        # Create final output dataframe with ALL test dates
        final_predictions = []
        for date in test_dates_original:
            if date in date_map:
                final_predictions.append(date_map[date])
            else:
                # For dates without predictions (should be rare), use the closest available prediction
                closest_date = min(date_map.keys(), key=lambda x: abs(x - date))
                final_predictions.append(date_map[closest_date])
        
        output_df = pd.DataFrame({
            'Date': test_dates_original,
            'Target': final_predictions
        })
        
        # Handle any remaining NaNs in final output
        if output_df['Target'].isna().any():
            print(f"Warning: {output_df['Target'].isna().sum()} NaN values in final predictions, replacing with median")
            output_df['Target'] = output_df['Target'].fillna(output_df['Target'].median())
        
        print(f"Final prediction count: {len(output_df)} (should match test data count: {len(test_df)})")
        
        # Save predictions to CSV in the Kaggle working directory
        output_path = '/kaggle/working/predictions.csv'
        output_df.to_csv(output_path, index=False)
        print(f"\nPredictions saved to {output_path}")
        
        print("\nSample of predictions:")
        print(output_df.head())
        
        return trained_models, output_df
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
