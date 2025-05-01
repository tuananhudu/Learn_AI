from pydantic import BaseModel 

class ApartmentRequest(BaseModel):
    MedInc: float
    HouseAge: int
    AveRooms: float
    AveBedrms: float
    Population: int
    AveOccup: float
    Latitude: float
    Longitude: float

class ApartmentResponse(BaseModel):
    price : float 
