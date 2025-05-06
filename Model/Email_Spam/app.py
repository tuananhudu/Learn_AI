from fastapi import FastAPI 
from domain.domain import EmailRequest , EmailResponse 
from services.email_service import EmailService 
app = FastAPI()

@app.post("/predict")
async def predict_email(request : EmailRequest) -> EmailResponse:
    return EmailService().predict(request=request)