from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import MinMaxScaler , StandardScaler , OneHotEncoder 

class PreProcessing : 
    def __init__(
            self , 
            mms_cols = None ,
            sds_cols = None ,
            ohe_cols = None , 
    ):
        self.mms_cols = mms_cols or []
        self.sds_cols = sds_cols or []
        self.ohe_cols = ohe_cols or []
    def build(self):
        step = [
            ('std' , StandardScaler() , self.sds_cols),
            ('mms' , MinMaxScaler() , self.mms_cols), 
            ('ohe' , OneHotEncoder() , self.ohe_cols)
        ]
        return ColumnTransformer(step)
    