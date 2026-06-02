from pydantic import BaseModel, Field, field_validator

class TextRequest(BaseModel):
    """
    Schema for validating the input text request.
    Automatically strips whitespace and ensures non-empty text.
    """
    text: str = Field(..., min_length=3, max_length=500)

    @field_validator("text")
    @classmethod
    def validate_and_strip_text(cls, v: str) -> str:
        # Strip leading and trailing whitespace characters
        cleaned = v.strip()
        # Verify length of the cleaned string
        if len(cleaned) < 3:
            raise ValueError("Text must contain at least 3 non-whitespace characters.")
        return cleaned