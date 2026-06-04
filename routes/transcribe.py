import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

router = APIRouter()

@router.post("")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Receives an audio file (e.g. from the client's MediaRecorder)
    and passes it to OpenAI's Whisper API to transcribe the text.
    """
    try:
        # Read the file bytes
        file_bytes = await file.read()
        
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")
            
        # Whisper requires a filename with a valid extension (e.g., .webm or .wav)
        # We can construct a tuple (filename, file_bytes) to pass to the client
        # The filename extension tells Whisper how to decode it
        filename = file.filename if file.filename else "audio.webm"
        if not filename.endswith((".webm", ".mp4", ".mp3", ".wav", ".m4a", ".ogg")):
            filename = "audio.webm" # Default assumption for MediaRecorder
            
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=(filename, file_bytes),
            # Optional parameters for better punctuation and style alignment
            prompt="Consulting case interview answer. Expected terms: EBITDA, CAGR, profitability, revenues, fixed costs, variable costs, market size, competitors."
        )
        
        return {"text": transcription.text}
        
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to transcribe audio: {str(e)}")
