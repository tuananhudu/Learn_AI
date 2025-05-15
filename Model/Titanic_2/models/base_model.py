from abc import ABC , abstractmethod
from typing import Any 

class BaseModel(ABC):
    @abstractmethod
    def fit(self , X : Any , y : Any) -> Any : 
        pass
    @abstractmethod
    def predict(self, X: Any) -> Any:
        pass

    @abstractmethod
    def score(self, X: Any, y: Any) -> float:
        pass