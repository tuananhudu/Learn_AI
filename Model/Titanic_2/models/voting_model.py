from typing import List, Tuple, Optional
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score , f1_score , classification_report , confusion_matrix 

from models.base_model import BaseModel
from preprocessing.preprocessor import Preprocessor

class VotingModel(BaseModel):
    def __init__(
            self , 
            preprocessor : Preprocessor , 
            models: Optional[List[Tuple[str, object]]] = None,
            param_grid: Optional[dict] = None,
            cv: int = 5
    ):
        self.preprocessor = preprocessor 
        self.models = models or [
            ('rf', RandomForestClassifier(random_state=42)),
            ('gb', GradientBoostingClassifier(random_state=42)),
            ('knn', KNeighborsClassifier()),
            ('logis' ,LogisticRegression() )
        ]
        self.param_grid = param_grid or {
            'voting__rf__n_estimators': [50, 100],
            'voting__rf__max_depth': [None, 10],
            'voting__knn__n_neighbors': [3, 5]
        }
        self.cv = cv
        self.pipeline = None
        self.grid_search = None
    def build_pipeline(self):
        voting = VotingClassifier(estimators=self.models , voting = 'hard')
        return Pipeline([
            ('preprocessor', self.preprocessor.build()),
            ('voting', voting)
        ])
    def fit(self, X, y):
        self.pipeline = self.build_pipeline()
        self.grid_search = GridSearchCV(self.pipeline, self.param_grid, cv=self.cv, scoring='r2', n_jobs=-1)
        self.grid_search.fit(X, y)
        return self
    def predict(self, X):
        return self.grid_search.predict(X)
    
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
    
    def report(self , X , y) : 
        return classification_report(y , self.predict(X))
    
    def cm(self , X , y):
        return confusion_matrix(y , self.predict(X))
    
    def best_params(self):
        return self.grid_search.best_params_
    