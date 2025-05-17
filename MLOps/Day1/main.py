import mlflow
from mlflow.models import infer_signature 

import pandas as pd 
import seaborn as sns 
from sklearn import datasets  # <-- THÊM DÒNG NÀY
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , precision_score , recall_score , f1_score

X , y = datasets.load_iris(return_X_y=True)

X_train , X_test , y_train , y_test = train_test_split(
    X , y , test_size=0.2 , random_state=42
)

params = {
    "solver": "lbfgs",
    "max_iter": 1000 , 
    "multi_class": "auto",
    "random_state": 8888,
}

lr = LogisticRegression(**params)
lr.fit(X_train , y_train)

y_pred = lr.predict(X_test)

accuracy = accuracy_score(y_test , y_pred)
precision = precision_score(y_test , y_pred, average='macro')  # <-- ADD average
recall = recall_score(y_test , y_pred, average='macro')        # <-- ADD average
f1 = f1_score(y_test , y_pred, average='macro')                # <-- ADD average

mlflow.set_tracking_uri(uri="http://localhost:5000")

# Create a new Mlflow Experiment 
mlflow.set_experiment("MLflow Quickstart")

# Start an Mlflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1 score", f1)

    # Set a tag to describe the run
    mlflow.set_tag("Training Info", "Basic LR model for iris data")

    # Infer the model signature
    signature = infer_signature(X_train, lr.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="iris_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-quickstart",
    )

# Load the model back for predictions
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
predictions = loaded_model.predict(X_test)

iris_feature_name = datasets.load_iris().feature_names

result = pd.DataFrame(X_test , columns=iris_feature_name)
result['actual_class'] = y_test
result['predicted_class'] = predictions

result.head(4)
