import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 🟡 1. Load dữ liệu Iris (dataset mẫu)
iris = datasets.load_iris()
X, y = iris.data, iris.target

# 🟡 2. Chia train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🟡 3. Khởi tạo model Logistic Regression với tham số
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}
model = LogisticRegression(**params)

# 🟡 4. Train model
model.fit(X_train, y_train)

# 🟡 5. Dự đoán và tính metrics
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# 🟡 6. Cấu hình MLflow tracking server
mlflow.set_tracking_uri(uri="http://localhost:5000")
mlflow.set_experiment("MLflow Iris Example1")

# 🟡 7. Bắt đầu 1 run để log tracking
with mlflow.start_run():
    # Log các hyperparameters
    mlflow.log_params(params)

    # Log các metrics kết quả
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Thêm tag mô tả
    mlflow.set_tag("Description", "Iris Classification with Logistic Regression")

    # Infer signature (schema input/output)
    signature = infer_signature(X_train, model.predict(X_train))

    # 🟡 8. Log model + Đăng ký vào Registry (registered_model_name)
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="iris_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="Iris-Classifier-Registry",
    )

print(f"Model version registered: {model_info.version}")

# 🟡 9. Load model từ Registry (bản mới nhất)
loaded_model = mlflow.pyfunc.load_model(f"models:/Iris-Classifier-Registry/{model_info.version}")

# 🟡 10. Predict lại từ model Registry
predictions = loaded_model.predict(X_test)

# 🟡 11. In kết quả thực tế vs dự đoán
result_df = pd.DataFrame(X_test, columns=iris.feature_names)
result_df["Actual Class"] = y_test
result_df["Predicted Class"] = predictions

print(result_df.head(5))

# from mlflow.tracking import MlflowClient

# client = MlflowClient()

# model_name = "Iris-Classifier-Registry"
# version = 3  # version bạn muốn chuyển stage
# new_stage = "Production"  # các giá trị khác: "Staging", "Archived", "None"

# client.transition_model_version_stage(
#     name=model_name,
#     version=version,
#     stage=new_stage,
#     archive_existing_versions=True  # Tự động chuyển các version khác về Archived
# )

# print(f"Model {model_name} version {version} đã chuyển sang stage {new_stage}")

# from fastapi import FastAPI
# from pydantic import BaseModel
# import mlflow.pyfunc
# import pandas as pd

# app = FastAPI()

# model_name = "Iris-Classifier-Registry"
# model_stage = "Production"
# model_uri = f"models:/{model_name}/{model_stage}"
# model = mlflow.pyfunc.load_model(model_uri)

# class InputData(BaseModel):
#     data: list[list[float]]  # dữ liệu dạng list of list số thực

# @app.post("/predict")
# async def predict(input_data: InputData):
#     df = pd.DataFrame(input_data.data)
#     preds = model.predict(df)
#     return {"predictions": preds.tolist()}

# from flask import Flask, request, jsonify
# import mlflow.pyfunc
# import pandas as pd

# app = Flask(__name__)

# # Load model từ registry, ví dụ version production
# model_name = "Iris-Classifier-Registry"
# model_stage = "Production"
# model_uri = f"models:/{model_name}/{model_stage}"

# model = mlflow.pyfunc.load_model(model_uri)

# @app.route("/predict", methods=["POST"])
# def predict():
#     json_data = request.json
#     # Giả sử dữ liệu gửi lên là list of features
#     data = pd.DataFrame(json_data["data"])
#     preds = model.predict(data)
#     return jsonify(predictions=preds.tolist())

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5001)
