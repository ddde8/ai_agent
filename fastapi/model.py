from pydantic import BaseModel

class Todo(BaseModel):
    id: int
    item: str

class Item(BaseModel):
    item: str
    status: str

class Todo(BaseModel):
    id: int
    item: Item