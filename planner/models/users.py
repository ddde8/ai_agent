from pydantic import BaseModel, EmailStr
from typing import Optional, List
from models.events import Event


class User(BaseModel):
    email: str
    password: str
    username: str
    events: Optional[List[str]] = None  # <- Now optional


    class Config:
        schema_extra = {
            "example": {
            "email": "fastapi@packt.com",
            "password": "strong!!!",
            "events": [],
        }
    }
        
class UserSignIn(BaseModel):
    email: EmailStr
    password: str

    class Config:
        schema_extra = {
            "example": {
                "email": "fastapi@packt.com",
                "password": "strong!!!"
            }
        }