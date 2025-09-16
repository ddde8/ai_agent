from pydantic import BaseModel
from typing import Optional, List
from fastapi import Form

class Todo(BaseModel):
    id: Optional[int] = None  # Make sure to provide a default value
    item: str

    @classmethod
    def as_form(
        cls,
        item: str = Form(...)
    ):
        return cls(item=item)

class Item(BaseModel):
    item: str
    status: str

'''
class Todo(BaseModel):
    id: int
    item: Item

    class Config:
        schema_extra = {
            "example": {
                "id": 1,
                "item": "Example Schema!"
            }
        }
'''

class TodoItem(BaseModel):
    item: str

    class Config:
        schema_extra = {
            "example": {
                "item": "Read the next chapter of the book"
            }
        }

class TodoItems(BaseModel):
    todos: List[TodoItem]

    class Config:
        schema_extra = {
            "example" : {
                "todos" : [
                    {
                        "item": "Example schema 1!"
                    },
                    {
                        "item": "Example schema 2!"
                    }
                ]
            }
        }