import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight
from loguru import logger
import os

from src.models.transformer_model import TimeSeriesTransformer
from src.data.features import FeatureEngineer

class ModelTrainer:
    """
    Handles model training with TimeSeriesSplit and custom loss.
    """
    
    def __init__(self, 
                 window_size: int = 60,
                 batch_size: int = 32,
                 epochs: int = 50,
                 learning_rate: float = 1e-4):
        self.window_size = window_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = None
        self.history = None

    def create_sequences(self, features: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert array of features and targets into sequences."""
        X, y = [], []
        for i in range(len(features) - self.window_size):
            X.append(features[i : i + self.window_size])
            # Target is the label immediately after the window
            y.append(targets[i + self.window_size])
        return np.array(X), np.array(y)

    def weighted_categorical_crossentropy(self, weights):
        """Custom loss function with class weights."""
        weights = tf.constant(weights, dtype=tf.float32)
        
        def loss(y_true, y_pred):
            # Scale predictions so that the class probas of each sample sum to 1
            y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)
            # Clip to prevent NaN's and Inf's
            y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
            # Calculate cross entropy
            loss = y_true * tf.math.log(y_pred) * weights
            loss = -tf.reduce_sum(loss, -1)
            return loss
        
        return loss

    def train(self, data: pd.DataFrame, feature_engineer: FeatureEngineer, n_splits: int = 5):
        """
        Train the model using TimeSeriesSplit.
        """
        # Prepare data
        # Assume data is already processed but not scaled OR we handle scaling here
        # Best practice: Fit scaler on TRAIN splits only to avoid leakage.
        
        # We need the targets. Let's assume 'label' column exists or we generate it
        if 'label' not in data.columns:
            logger.info("Generating Triple Barrier labels...")
            data['label'] = feature_engineer.triple_barrier_labels(data)
        
        # Drop rows where label is NaN (end of series)
        data = data.dropna(subset=['label'])
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        features_col = feature_engineer.feature_columns
        X = data[features_col].values
        y = data['label'].values
        
        fold = 1
        for train_index, val_index in tscv.split(X):
            logger.info(f"Training Fold {fold}/{n_splits}")
            
            X_train_raw, X_val_raw = data.iloc[train_index][features_col], data.iloc[val_index][features_col]
            y_train, y_val = y[train_index], y[val_index]
            
            # Fit scaler on Train
            feature_engineer.fit(X_train_raw)
            X_train_scaled = feature_engineer.transform(X_train_raw).values
            X_val_scaled = feature_engineer.transform(X_val_raw).values
            
            # Create Sequences
            X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train)
            X_val_seq, y_val_seq = self.create_sequences(X_val_scaled, y_val)
            
            # Compute Class Weights
            # Ensure y is integer for compute_class_weight
            classes = np.unique(y_train_seq)
            weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_seq)
            class_weights = dict(zip(classes, weights))
            logger.info(f"Class Weights: {class_weights}")
            
            # One-hot encode targets for categorical crossentropy
            # Or use sparse_categorical_accuracy
            
            # Build Model
            input_shape = (self.window_size, X_train_seq.shape[2])
            
            # Re-initialize model for each fold or continue training? 
            # Usually TimeSeriesSplit implies retraining on growing window.
            self.model = TimeSeriesTransformer(
                input_shape=input_shape,
                num_classes=3,
                learning_rate=self.learning_rate
            ).model
            
            # Compile with weighted loss if needed, but Keras fit() supports class_weight
            self.model.compile(
                optimizer=optimizers.Adam(learning_rate=self.learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Callbacks
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=5, restore_best_weights=True
            )
            
            # Train
            self.history = self.model.fit(
                X_train_seq, y_train_seq,
                validation_data=(X_val_seq, y_val_seq),
                epochs=self.epochs,
                batch_size=self.batch_size,
                class_weight=class_weights,
                callbacks=[early_stopping],
                verbose=1
            )
            
            fold += 1
            
        logger.info("Training completed.")
        return self.model

    def save_model(self, path):
        if self.model:
            self.model.save(path)
            
    def load_model(self, path):
        self.model = models.load_model(path) # load custom objects if needed
