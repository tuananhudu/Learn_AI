from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

class SklearnModel:
    def __init__(self, model: BaseEstimator, preprocessing, param_grid=None):
        self.model = model
        self.preprocessing = preprocessing
        self.param_grid = param_grid
        self.pipeline = None

    def fit(self, X, y):
        steps = [
            ('preprocessing', self.preprocessing.build()),
            ('model', self.model)
        ]
        pipeline = Pipeline(steps)

        if self.param_grid:
            self.pipeline = GridSearchCV(pipeline, param_grid=self.param_grid, cv=5, scoring='accuracy')
        else:
            self.pipeline = pipeline

        self.pipeline.fit(X, y)

    def evaluate(self, X, y):
        pred = self.pipeline.predict(X)
        acc = accuracy_score(y, pred)
        report = classification_report(y, pred)
        cm = confusion_matrix(y, pred)

        return {
            "accuracy": acc,
            "classification_report": report,
            "confusion_matrix": cm
        }
