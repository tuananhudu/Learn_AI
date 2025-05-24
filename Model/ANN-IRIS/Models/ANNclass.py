from typing import Tuple
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class ANN1:
    def __init__(
        self,
        input_shape: Tuple[int],
        output_shape: int,                # Số lớp (classes)
        batch_size: int = 32,
        epochs: int = 500,
        learning_rate: float = 0.01
    ):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.build_model()
        self.compile()
        self.stopping()

    def build_model(self):
        inputs = tf.keras.layers.Input(self.input_shape)
        x = tf.keras.layers.Dense(units=16, activation='relu')(inputs)
        x = tf.keras.layers.Dense(units=16, activation='relu')(x)
        outputs = tf.keras.layers.Dense(units=self.output_shape, activation='softmax')(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return self

    def compile(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return self

    def stopping(self, min_delta=0.001, patience=20, restore_best_weights=True):
        self.early_stopping = tf.keras.callbacks.EarlyStopping(
            min_delta=min_delta,
            patience=patience,
            restore_best_weights=restore_best_weights
        )
        return self

    def fit(self, X, y, plot=True):
        print("epochs:", self.epochs, type(self.epochs))
        print("batch_size:", self.batch_size, type(self.batch_size))
        print("X dtype:", X.dtype)
        print("y dtype:", y.dtype)

        history = self.model.fit(
            x=X,
            y=y,
            validation_split=0.2,
            callbacks=[self.early_stopping],
            epochs=self.epochs,
            batch_size=self.batch_size
        )
        if plot:
            metrics = ["loss", "accuracy", "val_loss", "val_accuracy"]
            fig, axes = plt.subplots(2, 2, figsize=(20, 10))
            for y_i in range(2):
                for x_i in range(2):
                    metric = metrics[y_i * 2 + x_i]
                    if metric in history.history:
                        axe = axes[y_i, x_i]
                        axe.plot(history.history[metric], label=metric, color="blue", alpha=0.6)
                        axe.set_xlabel("Epochs")
                        axe.set_ylabel(metric)
                        axe.legend()
            plt.tight_layout()
            plt.show()
        return self

    def predict(self, X):
        return self.model.predict(X)
