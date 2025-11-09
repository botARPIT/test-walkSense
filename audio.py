#!/usr/bin/env python3
"""
Local Voice Assistant Pipeline
Audio File → Text (Whisper) → LLM (Qwen3) → Speech (TTS)
"""

import subprocess
import sys
from pathlib import Path

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class VoiceAssistant:
    def __init__(
        self,
        whisper_model="base",
        qwen_url="http://localhost:11434/api/generate",  # Ollama default
        qwen_model="qwen3:1.7b",
        tts_method="system",  # Options: "system", "pyttsx3", "coqui"
        device="auto"  # Options: "auto", "cuda", "cpu"
    ):
        self.whisper_model_name = whisper_model
        self.qwen_url = qwen_url
        self.qwen_model = qwen_model
        self.tts_method = tts_method
        self.device = device
        
        # Initialize ASR
        self._init_asr()
        
        # Initialize TTS
        self._init_tts()
    
    def _init_asr(self):
        """Initialize Automatic Speech Recognition"""
        # Determine device
        device_to_use = self._get_device()
        
            
        if WHISPER_AVAILABLE:
            print(f"[ASR] Loading Whisper model: {self.whisper_model_name}")
            try:
                import torch
                # Load model with specified device
                self.asr_model = whisper.load_model(
                    self.whisper_model_name, 
                    device=device_to_use
                )
                print(f"[ASR] Using device: {device_to_use}")
                
                # Verify device
                actual_device = next(self.asr_model.parameters()).device
                print(f"[ASR] Model loaded on: {actual_device}")
                
            except Exception as e:
                print(f"[ASR] Error with {device_to_use}: {e}")
                print("[ASR] Falling back to CPU...")
                self.asr_model = whisper.load_model(self.whisper_model_name, device="cpu")
            
            self.asr_type = "whisper"
            
        
        elif FASTER_WHISPER_AVAILABLE:
            print(f"[ASR] Loading Faster-Whisper model: {self.whisper_model_name}")
            try:
                if device_to_use == "cuda":
                    # Try GPU with float16
                    self.asr_model = WhisperModel(
                        self.whisper_model_name,
                        device="cuda",
                        compute_type="float16"
                    )
                    print("[ASR] Using CUDA (float16)")
                else:
                    # Use CPU with int8 quantization
                    self.asr_model = WhisperModel(
                        self.whisper_model_name,
                        device="cpu",
                        compute_type="int8"
                    )
                    print("[ASR] Using CPU (int8)")
            except Exception as e:
                print(f"[ASR] GPU loading failed: {e}")
                print("[ASR] Falling back to CPU...")
                self.asr_model = WhisperModel(
                    self.whisper_model_name,
                    device="cpu",
                    compute_type="int8"
                )
            self.asr_type = "faster-whisper"
        else:
            raise ImportError(
                "Neither whisper nor faster-whisper is installed.\n"
                "Install with: pip install openai-whisper OR pip install faster-whisper"
            )
    
    def _get_device(self):
        """Determine which device to use"""
        if self.device == "cpu":
            return "cpu"
        elif self.device == "cuda":
            return "cuda"
        else:  # auto
            try:
                import torch
                if torch.cuda.is_available():
                    print("[INFO] CUDA available, will attempt GPU acceleration")
                    return "cuda"
                else:
                    print("[INFO] CUDA not available, using CPU")
                    return "cpu"
            except ImportError:
                print("[INFO] PyTorch not found, using CPU")
                return "cpu"
    
    def _init_tts(self):
        """Initialize Text-to-Speech"""
        if self.tts_method == "pyttsx3":
            try:
                import pyttsx3
                self.tts_engine = pyttsx3.init()
                # Configure voice properties
                self.tts_engine.setProperty('rate', 150)  # Speed
                self.tts_engine.setProperty('volume', 0.9)
                print("[TTS] Using pyttsx3")
            except ImportError:
                print("[TTS] pyttsx3 not available, falling back to system TTS")
                self.tts_method = "system"
        elif self.tts_method == "coqui":
            try:
                from TTS.api import TTS
                self.tts_engine = TTS("tts_models/en/ljspeech/tacotron2-DDC")
                print("[TTS] Using Coqui TTS")
            except ImportError:
                print("[TTS] Coqui TTS not available, falling back to system TTS")
                self.tts_method = "system"
        else:
            print("[TTS] Using system TTS commands")
    
    def transcribe_audio(self, audio_path):
        """Convert audio file to text using Whisper"""
        print(f"\n[1/4] Transcribing audio: {audio_path}")
        
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if self.asr_type == "faster-whisper":
            segments, info = self.asr_model.transcribe(
                audio_path, 
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            text = " ".join(segment.text for segment in segments)
        else:  # whisper
            # Force CPU and disable fp16 to avoid CUDA issues
            result = self.asr_model.transcribe(
                audio_path, 
                fp16=False,
                language="en"  # Specify language for faster processing
            )
            text = result["text"]
        
        text = text.strip()
        print(f"[ASR] Transcribed: '{text}'")
        return text
    
    def query_qwen(self, prompt):
        """Send prompt to local Qwen3 model"""
        print(f"\n[2/4] Querying Qwen3 model...")
        
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library not installed. Install with: pip install requests")
        
        # Prepare the payload for Ollama API
        payload = {
            "model": self.qwen_model,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(self.qwen_url, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            reply = result.get("response", "").strip()
            
            print(f"[LLM] Response: '{reply[:100]}{'...' if len(reply) > 100 else ''}'")
            return reply
            
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Failed to connect to Qwen3: {e}")
            print("\nTroubleshooting:")
            print("1. Is Ollama running? Start with: ollama serve")
            print("2. Is Qwen model loaded? Check with: ollama list")
            print(f"3. Try pulling the model: ollama pull {self.qwen_model}")
            return None
    
    def synthesize_speech(self, text, output_path="output.wav"):
        """Convert text to speech"""
        print(f"\n[3/4] Synthesizing speech...")
        
        if self.tts_method == "pyttsx3":
            self.tts_engine.save_to_file(text, output_path)
            self.tts_engine.runAndWait()
            print(f"[TTS] Audio saved to: {output_path}")
            return output_path
            
        elif self.tts_method == "coqui":
            self.tts_engine.tts_to_file(text=text, file_path=output_path)
            print(f"[TTS] Audio saved to: {output_path}")
            return output_path
            
        else:  # system TTS
            # Platform-specific TTS commands
            if sys.platform == "darwin":  # macOS
                subprocess.run(["say", text])
            elif sys.platform == "linux":
                # Try espeak or festival
                try:
                    subprocess.run(["espeak", text], check=True)
                except FileNotFoundError:
                    try:
                        subprocess.run(["festival", "--tts"], input=text.encode(), check=True)
                    except FileNotFoundError:
                        print("[TTS] No system TTS found. Install espeak: sudo apt-get install espeak")
            elif sys.platform == "win32":
                # Windows PowerShell TTS
                ps_command = f'Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{text}")'
                subprocess.run(["powershell", "-Command", ps_command])
            
            return None
    
    def play_audio(self, audio_path):
        """Play audio file"""
        print(f"\n[4/4] Playing audio: {audio_path}")
        
        if sys.platform == "darwin":  # macOS
            subprocess.run(["afplay", audio_path])
        elif sys.platform == "linux":
            # Try multiple players
            players = ["aplay", "paplay", "ffplay", "mpg123"]
            for player in players:
                try:
                    subprocess.run([player, audio_path], check=True)
                    break
                except FileNotFoundError:
                    continue
            else:
                print(f"[WARN] No audio player found. Audio saved to: {audio_path}")
        elif sys.platform == "win32":
            import os
            os.startfile(audio_path)
    
    def run_pipeline(self, audio_path, output_path="output.wav"):
        """Run the complete voice assistant pipeline"""
        print("="*60)
        print("VOICE ASSISTANT PIPELINE")
        print("="*60)
        
        # Step 1: Audio → Text
        transcribed_text = self.transcribe_audio(audio_path)
        
        if not transcribed_text:
            print("[ERROR] Transcription failed or empty")
            return None
        
        # Step 2: Text → LLM Response
        llm_response = self.query_qwen(transcribed_text)
        
        if not llm_response:
            print("[ERROR] LLM query failed")
            return None
        
        # Step 3: Text → Speech
        output_audio = self.synthesize_speech(llm_response, output_path)
        
        # Step 4: Play the audio (if file-based TTS)
        if output_audio and Path(output_audio).exists():
            self.play_audio(output_audio)
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED")
        print("="*60)
        
        return {
            "input_audio": audio_path,
            "transcribed_text": transcribed_text,
            "llm_response": llm_response,
            "output_audio": output_audio
        }


def main():
    """Example usage"""
    if len(sys.argv) < 2:
        print("Usage: python voice_assistant.py <audio_file.wav> [device]")
        print("\nExample: python voice_assistant.py input.wav")
        print("Example: python voice_assistant.py input.wav cpu")
        print("Example: python voice_assistant.py input.wav cuda")
        print("\nSupported formats: WAV, MP3, M4A, FLAC")
        print("\nDevice options:")
        print("  auto  - Automatically detect (default)")
        print("  cpu   - Force CPU usage (safe)")
        print("  cuda  - Force GPU usage (faster if available)")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    device = sys.argv[2] if len(sys.argv) > 2 else "auto"
    
    # Initialize the assistant
    assistant = VoiceAssistant(
        whisper_model="base",  # Options: tiny, base, small, medium, large
        qwen_url="http://localhost:11434/api/generate",  # Change if using different setup
        qwen_model="qwen3:1.7b",  # Change to your Qwen model name
        tts_method="system",  # Options: system, pyttsx3, coqui
        device=device  # auto, cpu, or cuda
    )
    
    # Run the pipeline
    result = assistant.run_pipeline(audio_file)
    
    if result:
        print(f"\n📝 Transcription: {result['transcribed_text']}")
        print(f"\n🤖 LLM Response: {result['llm_response']}")
        if result['output_audio']:
            print(f"\n🔊 Output Audio: {result['output_audio']}")


if __name__ == "__main__":
    main()