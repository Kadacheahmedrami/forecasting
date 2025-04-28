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
        self.scaler = RobustScaler()  # More robust to outliers
        self.target_scaler = RobustScaler()
        
    def preprocess_data(self, df):
        """Enhanced preprocessing with additional features"""
        # Make a copy of the dataframe to avoid modifying the original
        df = df.copy()
        
        # Convert date to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Add temporal features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Quarter'] = df['Date'].dt.quarter
        
        # Cyclic encoding of temporal features
        df['Month_sin'] = np.sin(2 * np.pi * df['Month']/12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month']/12)
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek']/7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek']/7)
        
        # Technical indicators
        df['MA7'] = df['Target'].rolling(window=7).mean()
        df['MA21'] = df['Target'].rolling(window=21).mean()
        df['MA50'] = df['Target'].rolling(window=50).mean()
        df['RSI'] = self._calculate_rsi(df['Target'])
        df['Volatility'] = df['Target'].rolling(window=7).std()
        
        # MACD (Moving Average Convergence Divergence)
        df['EMA12'] = df['Target'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Target'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Price momentum
        df['Price_Change'] = df['Target'].pct_change()
        df['Price_Change_3d'] = df['Target'].pct_change(periods=3)
        df['Price_Change_7d'] = df['Target'].pct_change(periods=7)
        
        # Target lag features
        for lag in [1, 2, 3, 5, 7]:
            df[f'Target_lag_{lag}'] = df['Target'].shift(lag)
        
        # A and B feature interactions - make sure these columns exist before calculating
        if 'A1' in df.columns and 'A2' in df.columns and 'A3' in df.columns and 'B1' in df.columns:
            # Add epsilon to avoid division by zero
            epsilon = 1e-8
            df['A1_A2_Ratio'] = df['A1'] / (df['A2'] + epsilon)
            df['A1_A3_Ratio'] = df['A1'] / (df['A3'] + epsilon)
            df['A2_A3_Ratio'] = df['A2'] / (df['A3'] + epsilon)
            df['A_B_Ratio'] = df['A1'] / (df['B1'] + epsilon)
            
            # Additional feature interactions
            df['A1_B1_Product'] = df['A1'] * df['B1']
            df['A2_B1_Product'] = df['A2'] * df['B1']
        
        # New features from action plan
        # Market regime features
        df['Volatility_21d'] = df['Target'].rolling(window=21).std()
        # Handle potential zeros in denominator
        df['Volatility_Ratio'] = df['Volatility'] / (df['Volatility_21d'] + 1e-8)
        
        # Trend strength indicators
        df['ADX'] = self._calculate_adx(df['Target'], period=14)
        
        # Add day-of-week dummy variables
        for i in range(5):  # For trading days Monday-Friday
            df[f'Day_{i}'] = (df['DayOfWeek'] == i).astype(int)
        
        # If volume data exists, add volume features
        if 'Volume' in df.columns:
            df['Volume_MA7'] = df['Volume'].rolling(window=7).mean()
            df['Volume_Change'] = df['Volume'].pct_change()
            df['Price_Volume_Ratio'] = df['Target'] / (df['Volume'] + 1e-8)
        
        # Replace NaN with forward fill then backward fill
        df = df.ffill().bfill()  # Updated to use recommended methods
        
        # Save original Target for later use
        original_target = df['Target'].copy()
        
        # Log transform the target (action plan item 1)
        # Make sure Target is positive before applying log transform
        if (df['Target'] <= 0).any():
            min_value = df['Target'].min()
            if min_value <= 0:
                # Shift values to make all positive
                df['Target'] = df['Target'] - min_value + 1e-3
                
        df['Target'] = np.log1p(df['Target'])
        
        # Scale features
        features = [col for col in df.columns if col not in ['Date', 'Target']]
        
        df[features] = self.scaler.fit_transform(df[features])
        df[['Target']] = self.target_scaler.fit_transform(df[['Target']])
        
        return df, original_target
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        # Separate gains and losses
        gains = delta.copy()
        losses = delta.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        # Calculate average gains and losses
        avg_gain = gains.rolling(window=period).mean()
        avg_loss = losses.rolling(window=period).mean()
        
        # Calculate RS with protection against division by zero
        epsilon = 1e-8
        rs = avg_gain / (avg_loss + epsilon)
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_adx(self, price, period=14):
        """Calculate Average Directional Index"""
        high = price.rolling(window=2).max()
        low = price.rolling(window=2).min()
        
        plus_dm = high.diff()
        minus_dm = low.diff(-1)
        plus_dm = plus_dm.copy()  # Make a copy to avoid SettingWithCopyWarning
        minus_dm = minus_dm.copy()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': (high - price.shift(1)).abs(),
            'lc': (low - price.shift(1)).abs()
        }).max(axis=1)
        
        # Smoothed averages
        atr = tr.rolling(window=period).mean()
        # Avoid division by zero
        epsilon = 1e-8
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / (atr + epsilon))
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / (atr + epsilon))
        
        # ADX calculation
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + epsilon))
        adx = dx.rolling(window=period).mean()
        return adx
    
    def create_sequences(self, df):
        """Create windowed sequences for time series prediction"""
        features = [col for col in df.columns if col not in ['Date']]
        
        X, y = [], []
        dates = []
        
        for i in range(len(df) - self.seq_length):
            X.append(df[features].iloc[i:(i + self.seq_length)].values)
            y.append(df['Target'].iloc[i + self.seq_length])
            dates.append(df['Date'].iloc[i + self.seq_length])
            
        return np.array(X), np.array(y), dates
    
    def build_model(self, input_shape):
        """Build an improved LSTM model with attention mechanisms"""
        # Fix for input shape issue
        if isinstance(input_shape, tuple) and len(input_shape) == 2:
            # If we got a tuple with two values, restructure it
            seq_length, features = input_shape
            input_shape = (seq_length, features)
        else:
            # Use the original 3D tensor shape
            input_shape = (input_shape[1], input_shape[2])
        
        # Input layer
        inputs = tf.keras.Input(shape=input_shape)
        
        # LSTM layers with attention
        x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
        x = Dropout(0.3)(x)
        
        # Add self-attention layer (Transformer component from action plan item 4)
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=32
        )(x, x)
        x = tf.keras.layers.Add()([x, attention])
        x = tf.keras.layers.LayerNormalization()(x)
        
        # Continue with LSTM layers
        x = Bidirectional(LSTM(64, return_sequences=True))(x)
        x = Dropout(0.3)(x)
        x = Bidirectional(LSTM(32))(x)
        x = Dropout(0.3)(x)
        
        # Dense layers
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(16, activation='relu')(x)
        outputs = Dense(1)(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Use custom MAPE loss (action plan item 2)
        model.compile(
            optimizer='adam', 
            loss=mape_loss,  
            metrics=['mae', 'mse', tf.keras.metrics.RootMeanSquaredError()]
        )
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the model with callbacks"""
        # Fix: Pass X_train.shape directly instead of creating a nested tuple
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
        """Evaluate model performance with expanded metrics"""
        y_pred = model.predict(X_test)
        
        # Inverse transform predictions and actual values (scaled)
        y_test_scaled_inv = self.target_scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_scaled_inv = self.target_scaler.inverse_transform(y_pred)
        
        # Inverse log transform (action plan item 1)
        y_test_inv = np.expm1(y_test_scaled_inv)
        y_pred_inv = np.expm1(y_pred_scaled_inv)
        
        # Calculate metrics
        mse = np.mean((y_test_inv - y_pred_inv) ** 2)
        mae = np.mean(np.abs(y_test_inv - y_pred_inv))
        rmse = np.sqrt(mse)
        
        # Calculate MAPE carefully to avoid division by zero
        epsilon = 1e-8
        abs_percentage_errors = np.abs((y_test_inv - y_pred_inv) / np.maximum(np.abs(y_test_inv), epsilon)) * 100
        mape = np.mean(abs_percentage_errors)
        
        # Calculate directional accuracy (prediction gets direction of movement correct)
        direction_actual = np.sign(np.diff(np.vstack([y_test_inv[0], y_test_inv]), axis=0))
        direction_pred = np.sign(np.diff(np.vstack([y_test_inv[0], y_pred_inv]), axis=0))
        # Ignore zeros in direction comparison
        dir_compare = direction_actual == direction_pred
        dir_compare = dir_compare[direction_actual != 0]
        if len(dir_compare) > 0:
            directional_accuracy = np.mean(dir_compare) * 100
        else:
            directional_accuracy = 0
            
        metrics = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Directional Accuracy': directional_accuracy
        }
        
        return metrics, y_pred_inv
    
    def plot_predictions(self, dates, y_true, y_pred, metrics=None, title='Stock Price Predictions'):
        """Improved visualization of predictions"""
        plt.figure(figsize=(15, 7))
        
        # Plot actual and predicted values
        plt.plot(dates, y_true, label='Actual', marker='o', markersize=3, alpha=0.7)
        plt.plot(dates, y_pred, label='Predicted', marker='x', markersize=3, alpha=0.7)
        
        # Add metrics to title if provided
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
        
        # Plot error distribution
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
        
        # Plot actual vs predicted scatter
        plt.figure(figsize=(10, 8))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        
        # Calculate correlation
        corr = np.corrcoef(y_true.flatten(), y_pred.flatten())[0,1]
        r2 = corr**2
        
        plt.title(f'Actual vs Predicted Values (R² = {r2:.4f})', fontsize=16)
        plt.xlabel('Actual', fontsize=12)
        plt.ylabel('Predicted', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    def analyze_feature_importance(self, X, feature_names):
        """Analyze feature importance using basic approach"""
        # Flatten the 3D data
        X_flat = X.reshape(X.shape[0], -1)
        
        # Calculate mean absolute values for each feature across time steps
        feature_importance = {}
        
        for i, feature in enumerate(feature_names):
            # Extract this feature across all time steps
            feature_data = X[:, :, i]
            importance = np.mean(np.abs(feature_data))
            feature_importance[feature] = importance
            
        # Sort by importance
        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Plot
        plt.figure(figsize=(12, 8))
        features = [x[0] for x in sorted_importance[:15]]  # Top 15 features
        values = [x[1] for x in sorted_importance[:15]]
        
        plt.barh(features, values)
        plt.xlabel('Mean Absolute Value')
        plt.title('Feature Importance Analysis')
        plt.gca().invert_yaxis()  # Highest at top
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
        # Load data - update path to match your file location
        train_df = pd.read_csv('/kaggle/input/data-bounty-1-stock-forcasting/Problem-1-train.csv')
        
        # Display basic information about the data
        print("Dataset Information:")
        print(f"Total rows: {len(train_df)}")
        print(f"Date range: {train_df['Date'].min()} to {train_df['Date'].max()}")
        
        # Check for missing values
        missing_values = train_df.isnull().sum()
        print("\nMissing Values:")
        print(missing_values)
        
        # Check if needed columns exist
        required_cols = ['Date', 'Target']
        if not all(col in train_df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in train_df.columns]
            raise ValueError(f"Required columns missing: {missing}")
        
        print("\nTraining heterogeneous ensemble...")
        # Train the heterogeneous ensemble (action plan item 5)
        ensemble = HeterogeneousEnsemble(seq_length=30)
        trained_models = ensemble.train(train_df)
        
        # Use the same preprocessing as in ensemble training
        processed_df, original_target = ensemble.lstm_forecaster.preprocess_data(train_df)
        X, y, dates = ensemble.lstm_forecaster.create_sequences(processed_df)
        
        # Split data for final evaluation
        split_idx = int(len(X) * 0.85)  # Use last 15% for final testing
        X_test, y_test = X[split_idx:], y[split_idx:]
        test_dates = dates[split_idx:]
        
        print(f"Training set size: {len(X) - len(X_test)}")
        print(f"Test set size: {len(X_test)}")
        
        # Generate ensemble predictions
        print("\nGenerating ensemble predictions...")
        ensemble_predictions = ensemble.predict(X_test, trained_models['lstm_model'])
        
        # Inverse transform predictions
        y_test_scaled_inv = ensemble.lstm_forecaster.target_scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_scaled_inv = ensemble.lstm_forecaster.target_scaler.inverse_transform(ensemble_predictions)
        
        # If log transform was applied, reverse it
        y_test_inv = np.expm1(y_test_scaled_inv)
        y_pred_inv = np.expm1(y_pred_scaled_inv)
        
        # Calculate and display metrics
        metrics = calculate_metrics(y_test_inv, y_pred_inv)
        print("\nEnsemble Model Performance:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        
        # Plot results
        ensemble.lstm_forecaster.plot_predictions(
            test_dates,
            y_test_inv,
            y_pred_inv,
            metrics=metrics,
            title='Heterogeneous Ensemble Predictions'
        )
        
        # For comparison, let's also see how the LSTM model alone performs
        print("\nGenerating LSTM-only predictions...")
        lstm_pred = trained_models['lstm_model'].predict(X_test)
        lstm_pred_scaled_inv = ensemble.lstm_forecaster.target_scaler.inverse_transform(lstm_pred)
        lstm_pred_inv = np.expm1(lstm_pred_scaled_inv)
        
        lstm_metrics = calculate_metrics(y_test_inv, lstm_pred_inv)
        print("\nLSTM Model Alone Performance:")
        for k, v in lstm_metrics.items():
            print(f"{k}: {v:.4f}")
        
        ensemble.lstm_forecaster.plot_predictions(
            test_dates,
            y_test_inv,
            lstm_pred_inv,
            metrics=lstm_metrics,
            title='LSTM Model Predictions'
        )
        
        # Feature importance analysis on the test set
        print("\nAnalyzing feature importance...")
        feature_names = [col for col in processed_df.columns if col not in ['Date', 'Target']]
        feature_importance = ensemble.lstm_forecaster.analyze_feature_importance(X_test, feature_names)
        
        print("\nTop 15 most important features:")
        for i, (feature, importance) in enumerate(feature_importance[:15]):
            print(f"{i+1}. {feature}: {importance:.6f}")
        
        return trained_models, y_test_inv, y_pred_inv, test_dates
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()