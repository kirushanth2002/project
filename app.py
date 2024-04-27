# whisperTransformer - LLama - translation

from flask import Flask, render_template, request, redirect
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
import librosa
import torch
from deep_translator import GoogleTranslator
app = Flask(__name__)
# Initialize the pipeline for text generation
text_gen_pipe = pipeline("text-generation", model="Kirushanth02/llama-2-7b")
# Load the Whisper model processor and model
processor = AutoProcessor.from_pretrained("openai/whisper-large-v2")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v2")
@app.route("/templates/index.html", methods=["GET", "POST"])
def index():
transcript = ""
corrected_transcript = ""
translate = ""
if request.method == "POST":
if "file" not in request.files:
return redirect(request.url)
file = request.files["file"]
language1 = request.form.get('languaget')
if file.filename == "":
return redirect(request.url)
if file:
try:
# Load the audio file
audio, rate = librosa.load(file, sr=16000)
# Process the audio file
input_values = processor(audio, sampling_rate=rate, return_tensors="pt").input_values
# Generate the transcript
with torch.no_grad():
outputs = model.generate(input_values)
transcript = processor.batch_decode(outputs, skip_special_tokens=True)[0]
# Use the pipeline to get the corrected transcript
corrected_transcript = text_gen_pipe(transcript)[0]['generated_text']
# Perform translation (existing logic)
if language1 == "Hindi":
translate = GoogleTranslator(source='english', target='hindi').translate(corrected_transcript)
elif language1 == "Bengali":
translate = GoogleTranslator(source='english', target='bengali').translate(corrected_transcript)
elif language1 == "Telugu":
translate = GoogleTranslator(source='english', target='telugu').translate(corrected_transcript)
elif language1 == "Tamil":
translate = GoogleTranslator(source='english', target='tamil').translate(corrected_transcript)
elif language1 == "Malayalam":
translate = GoogleTranslator(source='english', target='malayalam').translate(corrected_transcript)
else:
translate = "Oops! This particular language cannot be translated."
except Exception as e:
return f"An error occurred: {str(e)}"
return render_template('index.html', transcript=transcript, corrected_transcript=corrected_transcript, translate=translate)
if __name__ == "__main__":
app.run(debug=True, threaded=True) i need to run this code and get output help me