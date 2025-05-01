import pandas as pd
import numpy as np
from domain.domain import ApartmentRequest, ApartmentResponse
import joblib
from tensorflow.keras.models import load_model
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

class ApartmentService:
    def __init__(self):
        # Sử dụng đường dẫn tương đối để linh hoạt hơn
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.path_model = os.path.join(base_dir, '../artifacts/ann_housing_model.h5')
        self.path_scaler_X = os.path.join(base_dir, '../artifacts/scaler_X.pkl')
        self.path_scaler_y = os.path.join(base_dir, '../artifacts/scaler_y.pkl')
        
        self.model = load_model(self.path_model)  # Tải mô hình ANN
        self.scaler_X = joblib.load(self.path_scaler_X)  # Tải scaler X
        self.scaler_y = joblib.load(self.path_scaler_y)  # Tải scaler y

    def preprocess_input(self, request: ApartmentRequest) -> pd.DataFrame:
        data_dict = {
            "MedInc": request.MedInc,
            "HouseAge": request.HouseAge,
            "AveRooms": request.AveRooms,
            "AveBedrms": request.AveBedrms,
            "Population": request.Population,
            "AveOccup": request.AveOccup,
            "Latitude": request.Latitude,
            "Longitude": request.Longitude
        }
        data_df = pd.DataFrame([data_dict])
        data_df = self.scaler_X.transform(data_df.to_numpy())  # Chuẩn hóa dữ liệu
        return data_df

    def predict_price(self, request: ApartmentRequest) -> ApartmentResponse:
        input_df = self.preprocess_input(request)
        
        apartment_price = self.model.predict(input_df)[0]  
        apartment_price = self.scaler_y.inverse_transform([apartment_price])[0][0]  
        apartment_price = max(0, apartment_price)
        response = ApartmentResponse(price=float(apartment_price))  # Chuyển sang float
        return response
    
# if __name__ =='__main__':
#     test_request = ApartmentRequest(
#     MedInc=5.2,
#     HouseAge=20,
#     AveRooms=5.8,
#     AveBedrms=1.0,
#     Population=1200,
#     AveOccup=3.0,
#     Latitude=34.15,
#     Longitude=-118.35
# )
#     apt_serv = ApartmentService()
#     res = apt_serv.predict_price(request=test_request)
#     print(res.price)
