import soundfile as sf
import whisper

audio_file_path = "./test.mp3"
audio_file, sr = sf.read(audio_file_path)
audio_model = whisper.load_model("base")
result = audio_model.transcribe(audio_file_path)
transcription = result["text"]
print("Transcription", transcription)
