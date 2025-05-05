# from typing import *
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import numpy as np

# class SoftmaxRegression:
#     def __init__(
#             self,
#             input_shape: tuple,
#             output_shape: int = 1,  # 1 cho phân loại nhị phân
#             learning_rate: float = 0.001,  # Giảm learning rate cho Adam
#             epochs: int = 50,  # Giảm số epoch để thử nghiệm nhanh
#             batch_size: int = 32
#     ):
#         self.input_shape = input_shape
#         self.output_shape = output_shape
#         self.learning_rate = learning_rate
#         self.epochs = epochs
#         self.batch_size = batch_size

#         self._build_model()

#     def _build_model(self):
#         # Định nghĩa mô hình với lớp ẩn và Dropout
#         inputs = tf.keras.layers.Input(shape=self.input_shape)
#         x = tf.keras.layers.Dense(128, activation="relu")(inputs)  # Lớp ẩn 1
#         x = tf.keras.layers.Dropout(0.3)(x)  # Dropout để giảm overfitting
#         x = tf.keras.layers.Dense(64, activation="relu")(x)  # Lớp ẩn 2
#         x = tf.keras.layers.Dropout(0.3)(x)
#         outputs = tf.keras.layers.Dense(units=self.output_shape, activation="sigmoid")(x)

#         self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
#         self.model.compile(
#             optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),  # Dùng Adam
#             loss="binary_crossentropy",  # Phù hợp với nhãn [0, 1] hoặc [True, False]
#             metrics=["accuracy"]
#         )

#         self.model.summary()
#         return self

#     def _compute_class_weights(self, y):
#         n_samples = len(y)
#         n_class_0 = np.sum(y == 0)  # Số lượng False (0)
#         n_class_1 = np.sum(y == 1)  # Số lượng True (1)

#         # Tính trọng số
#         weight_0 = (1 / n_class_0) * (n_samples / 2.0) if n_class_0 > 0 else 1.0
#         weight_1 = (1 / n_class_1) * (n_samples / 2.0) if n_class_1 > 0 else 1.0

#         return {0: weight_0, 1: weight_1}

#     def fit(self, X, y, verbose=1, plot_history=True):

#         # Chuyển y thành dạng số (True/False thành 1/0)
#         y = y.astype(int)  # Chuyển True/False thành 1/0
#         y = np.squeeze(y)  # Đảm bảo y là (n_samples,)

#         # Trộn dữ liệu
#         indices = np.arange(X.shape[0])
#         np.random.shuffle(indices)
#         X = X[indices]
#         y = y[indices]

#         # Tính class weights
#         class_weights = self._compute_class_weights(y)
#         print("Class weights:", class_weights)

#         history = self.model.fit(
#             x=X,
#             y=y,
#             validation_split=0.2,
#             epochs=self.epochs,
#             batch_size=self.batch_size,
#             verbose=verbose,
#             class_weight=class_weights
#         )

#         if plot_history:
#             self._plot_history(history)

#         return self

#     def _plot_history(self, history):
#         """
#         Vẽ biểu đồ lịch sử huấn luyện.
#         """
#         plt.figure(figsize=(12, 4))

#         # Độ chính xác
#         plt.subplot(1, 2, 1)
#         plt.plot(history.history['accuracy'], label='Training Accuracy')
#         plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
#         plt.title('Model Accuracy')
#         plt.xlabel('Epoch')
#         plt.ylabel('Accuracy')
#         plt.legend()

#         # Hàm mất mát
#         plt.subplot(1, 2, 2)
#         plt.plot(history.history['loss'], label='Training Loss')
#         plt.plot(history.history['val_loss'], label='Validation Loss')
#         plt.title('Model Loss')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.legend()

#         plt.tight_layout()
#         plt.show()

#     def predict(self, X):
#         probs = self.model.predict(X)
#         return (probs > 0.5).astype(int).flatten()  # Dự đoán nhị phân