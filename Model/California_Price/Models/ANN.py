from typing import *
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from regression_metrics.eval_metrics import RegressionMetrics

class ANN:
    def __init__(
        self,
        input_shape: Tuple[int, int],
        output_shape: int,
        batch_size: int = 32,
        epochs: int = 500,
        learning_rate: float = 0.01
    ):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.build_model()  # Gọi phương thức build_model để tạo self.model
        self.compile()      # Biên dịch mô hình
        self.stopping()     # Thiết lập early stopping

    def build_model(self):
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        x = tf.keras.layers.Dense(units=8, activation='relu')(inputs)
        x = tf.keras.layers.Dense(units=8, activation='relu')(x)
        outputs = tf.keras.layers.Dense(units=self.output_shape)(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)  # Khởi tạo self.model
        return self

    def compile(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate),
            loss='mse',
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )
        self.model.summary()
        return self

    def stopping(self, min_delta=0.001, patience=20, restore_best_weights=True):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            min_delta=min_delta,
            patience=patience,
            restore_best_weights=restore_best_weights
        )
        self.early_stopping = early_stopping
        return self

    def fit(self, X, y, plot=True):
        if X.shape[1:] != self.input_shape:
            raise ValueError(f"Input shape mismatch: Expected {self.input_shape}, got {X.shape[1:]}")
        
        history = self.model.fit(
            x=X,
            y=y,
            validation_split=0.2,
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=[self.early_stopping]
        )

        if plot:
            metrics = ["loss", "root_mean_squared_error", "val_loss", "val_root_mean_squared_error"]
            fig, axes = plt.subplots(2, 2, figsize=(20, 10))
            for y in range(2):
                for x in range(2):
                    axe = axes[y, x]
                    metric = metrics[y * 2 + x]
                    axe.plot(history.history[metrics[y * 2 + x]], label = metrics[y * 2 + x], color = "blue", alpha = 0.4, linestyle = "-")
                    axe.set_xlabel("Epochs")
                    axe.set_ylabel(metric)
                    axe.legend()
            plt.tight_layout()
            plt.show()

        return self

    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate_custom_metrics(self, X_test, y_test, inverse_transform=False):
        y_pred = self.predict(X_test)
        if inverse_transform and hasattr(self, "y_scaler"):
            y_pred = self.y_scaler.inverse_transform(y_pred)
            y_test = np.array(y_test).reshape(y_pred.shape)
        metrics = RegressionMetrics(y_pred=y_pred, y_true=y_test)
        result = metrics.mse().mae().rmse().r2_score().summary()
        print("== Custom Regression Metrics ==")
        for k, v in result.items():
            print(f"{k.upper()}: {v:.4f}")
        return result
