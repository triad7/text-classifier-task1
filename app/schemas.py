from pydantic import BaseModel, Field

class TextRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=500)