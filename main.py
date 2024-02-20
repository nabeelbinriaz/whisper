from fastapi import FastAPI, File, UploadFile,Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import whisper
import tempfile
from transformers import AutoTokenizer, AutoModelWithLMHead
import torch
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

app = FastAPI()
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
# Function to transcribe audio using Whisper
async def transcribe_audio(audio_path):
    model = whisper.load_model("large")  # Adjust model size as needed
    result = model.transcribe(audio_path,fp16=False)
    return result["text"]

@app.post("/transcribe/")
async def transcribe_audio_file(file: UploadFile = File(...)):
    try:
        # Save uploaded audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
            tmp_audio.write(await file.read())
            audio_path = tmp_audio.name

        # Transcribe audio to text
        text = await transcribe_audio(audio_path)

        return JSONResponse(status_code=200, content={"transcription": text})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.post("/summarize/")
async def summarzie_audio_file(text: str = Form(...)):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    m_name = "marefa-nlp/summarization-arabic-english-news"

    tokenizer = AutoTokenizer.from_pretrained(m_name)
    model = AutoModelWithLMHead.from_pretrained(m_name).to(device)

    def get_summary(text, tokenizer, model, device="cpu", num_beams=8, max_length=2048, length_penalty=5.0):
        if len(text.strip()) < 50:
            return ["Please provide a longer text."]

        text = "summarize in detail: <paragraph> " + " <paragraph> ".join([s.strip() for s in sent_tokenize(text) if s.strip() != ""]) + " </s>"
        text = text.strip().replace("\n","")

        tokenized_text = tokenizer.encode(text, return_tensors="pt").to(device)

        summary_ids = model.generate(
                tokenized_text,
                max_length=max_length,
                num_beams=num_beams,
                repetition_penalty=1.5,
                length_penalty=length_penalty,
                early_stopping=True
        )

        output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return [s.strip() for s in output.split("<hl>") if s.strip() != ""]
    hls = get_summary(text, tokenizer, model, device)
    a=""
    for hl in hls:
        summary=" "+a+hl+" "
    return JSONResponse(status_code=200, content={"summary": summary})
