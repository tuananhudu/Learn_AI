from sklearn.base import BaseEstimator
from models.base_model import BaseModel 
from sklearn.metrics import accuracy_score

class SklearnModel(BaseModel):
    def __init__(self, model: BaseEstimator):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        preds = self.predict(X)
        return accuracy_score(y, preds)
