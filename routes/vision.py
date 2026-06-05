import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

router = APIRouter()

class ExtractTextRequest(BaseModel):
    base64_image: str

@router.post("")
async def extract_text(request: ExtractTextRequest):
    """
    Receives a base64 encoded image string (data:image/jpeg;base64,...) 
    and passes it to OpenAI's GPT-4o vision capabilities to extract handwritten/printed text.
    """
    try:
        if not request.base64_image:
            raise HTTPException(status_code=400, detail="Empty image data")
            
        # Ensure it has the data URI scheme if not provided
        base64_data = request.base64_image
        if not base64_data.startswith("data:image"):
            # Assume jpeg if no prefix
            base64_data = f"data:image/jpeg;base64,{base64_data}"

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert OCR system. Extract all handwritten or printed text from this image exactly as written. Preserve structural elements like bullet points, numbering, and indentation. Do not add any conversational filler or introductions; return ONLY the extracted text."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": base64_data
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        extracted_text = response.choices[0].message.content
        return {"text": extracted_text}
        
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extract text: {str(e)}")
