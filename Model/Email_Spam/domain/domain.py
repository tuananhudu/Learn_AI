from pydantic import BaseModel
from typing import Optional, Literal

class EmailRequest(BaseModel):
    message: Optional[str] = ""  

class EmailResponse(BaseModel):
    reply: Literal[0, 1] 