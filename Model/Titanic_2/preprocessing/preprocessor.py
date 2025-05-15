from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler , MinMaxScaler , OneHotEncoder

class Preprocessor : 
    def __init__(self , standard_cols = None , minmax_cols = None , categorical_cols = None) :
        self.standard_cols = standard_cols or []
        self.minmax_cols = minmax_cols or []
        self.categorical_cols = categorical_cols or []

    def build(self):
        return ColumnTransformer([
            ('std' , StandardScaler() , self.standard_cols) , 
            ('minmax' , MinMaxScaler() , self.minmax_cols) , 
            ('cat' , OneHotEncoder(handle_unknown='ignore') , self.categorical_cols)
        ])
    