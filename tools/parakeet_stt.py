#!/usr/bin/env python3
"""
NVIDIA Parakeet TDT 0.6B V2 Speech-to-Text tool.

Uses NVIDIA NeMo toolkit to load and use Parakeet model from .nemo checkpoint file.
Supports both audio file transcription and real-time microphone input.
"""
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import nemo.collections.asr as nemo_asr
    import torch
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    torch = None


def capture_audio_arecord(card=1, device=0, duration=5, sample_rate=16000, channels=1):
    """
    Capture audio using arecord command (for USB lavalier microphone).
    
    Args:
        card: ALSA card number (default: 1 for USB Lavalier Microphone)
        device: ALSA device number (usually 0)
        duration: Recording duration in seconds
        sample_rate: Sample rate (16kHz required for Parakeet)
        channels: Number of audio channels (1 = mono)
    
    Returns:
        Path to temporary WAV file, or None on error
    """
    # Create temporary WAV file
    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_wav_path = temp_wav.name
    temp_wav.close()
    
    # Build arecord command
    # Note: arecord requires integer duration
    duration_int = int(round(duration))
    cmd = [
        'arecord',
        '-D', f'hw:{card},{device}',
        '-f', 'S16_LE',  # 16-bit signed little-endian
        '-r', str(sample_rate),
        '-c', str(channels),
        '-d', str(duration_int),
        temp_wav_path
    ]
    
    try:
        # Run arecord
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for process to finish
        stdout, stderr = process.communicate(timeout=duration + 2)
        
        if process.returncode != 0:
            print(f"[ERROR] Audio recording failed: {stderr}")
            try:
                Path(temp_wav_path).unlink()
            except:
                pass
            return None
        
        return temp_wav_path
        
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        print("[ERROR] Audio recording timed out")
        try:
            Path(temp_wav_path).unlink()
        except:
            pass
        return None
    except Exception as e:
        print(f"[ERROR] Recording failed: {e}")
        try:
            Path(temp_wav_path).unlink()
        except:
            pass
        return None


class ParakeetSTT:
    """NVIDIA Parakeet TDT 0.6B V2 Speech-to-Text wrapper."""
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize Parakeet STT model.
        
        Args:
            model_path: Path to .nemo checkpoint file. If None, tries to find
                       model in models/parakeet/ directory or loads from HuggingFace.
            device: Device to use ('cuda', 'cpu', or None for auto-detect).
                    If None, will try CUDA if available, otherwise CPU.
        """
        if not NEMO_AVAILABLE:
            raise ImportError(
                "NVIDIA NeMo toolkit not installed. Install with:\n"
                "  pip install -U nemo_toolkit[\"asr\"]"
            )
        
        self.model = None
        self.model_path = model_path
        
        # Determine device
        if device is None:
            # Auto-detect: try CUDA if available
            if torch is not None and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device.lower()
        
        # Validate device choice
        if self.device == "cuda" and (torch is None or not torch.cuda.is_available()):
            print(f"[WARNING] CUDA requested but not available. Using CPU instead.")
            self.device = "cpu"
        
        # Try to find model if path not provided
        if model_path is None:
            # Check local models directory first
            local_model_paths = [
                PROJECT_ROOT / "models" / "parakeet" / "parakeet-tdt-0.6b-v2.nemo",
                PROJECT_ROOT / "models" / "parakeet" / "*.nemo",
            ]
            
            # Try exact path first
            if local_model_paths[0].exists():
                model_path = str(local_model_paths[0])
            else:
                # Try to find any .nemo file in parakeet directory
                parakeet_dir = PROJECT_ROOT / "models" / "parakeet"
                if parakeet_dir.exists():
                    nemo_files = list(parakeet_dir.glob("*.nemo"))
                    if nemo_files:
                        model_path = str(nemo_files[0])
                    else:
                        # Fallback: try loading from HuggingFace
                        model_path = "nvidia/parakeet-tdt-0.6b-v2"
                else:
                    # Fallback: try loading from HuggingFace
                    model_path = "nvidia/parakeet-tdt-0.6b-v2"
        
        self.model_path = model_path
        self._load_model()
    
    def _load_model(self):
        """Load the Parakeet model from checkpoint or HuggingFace."""
        try:
            print(f"[ParakeetSTT] Loading model (device: {self.device})...")
            
            if self.model_path.endswith('.nemo'):
                # Load from local .nemo checkpoint file
                print(f"[ParakeetSTT] Loading model from: {self.model_path}")
                self.model = nemo_asr.models.ASRModel.restore_from(self.model_path)
            else:
                # Load from HuggingFace
                print(f"[ParakeetSTT] Loading model from HuggingFace: {self.model_path}")
                self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.model_path)
            
            # Move model to specified device
            if self.device == "cuda" and torch is not None and torch.cuda.is_available():
                print(f"[ParakeetSTT] Moving model to CUDA device...")
                self.model = self.model.to(torch.device("cuda"))
            else:
                print(f"[ParakeetSTT] Using CPU device")
                self.model = self.model.to(torch.device("cpu"))
            
            print("[ParakeetSTT] Model loaded successfully")
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Parakeet model from {self.model_path}: {e}\n"
                "Make sure:\n"
                "  1. NeMo toolkit is installed: pip install -U nemo_toolkit[\"asr\"]\n"
                "  2. Model file exists at the specified path\n"
                "  3. You have a HuggingFace account and token if loading from HuggingFace"
            ) from e
    
    def transcribe(
        self,
        audio_path: str,
        timestamps: bool = False,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio file using Parakeet model.
        
        Args:
            audio_path: Path to audio file (WAV or FLAC, 16kHz mono)
            timestamps: Whether to return timestamps (word/segment/char level)
            language: Optional language code (usually auto-detected)
        
        Returns:
            Dictionary with transcription and optional timestamps:
            {
                "text": str,
                "timestamps": {
                    "word": List[Dict],
                    "segment": List[Dict],
                    "char": List[Dict]
                } (if timestamps=True)
            }
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        try:
            # Transcribe audio
            if timestamps:
                output = self.model.transcribe([audio_path], timestamps=True)
            else:
                output = self.model.transcribe([audio_path], timestamps=False)
            
            # Extract transcription
            if isinstance(output, list) and len(output) > 0:
                result = output[0]
                
                # Format response
                response = {
                    "text": result.text if hasattr(result, 'text') else str(result)
                }
                
                # Add timestamps if requested
                if timestamps and hasattr(result, 'timestamp'):
                    response["timestamps"] = {
                        "word": result.timestamp.get('word', []),
                        "segment": result.timestamp.get('segment', []),
                        "char": result.timestamp.get('char', [])
                    }
                
                return response
            else:
                return {"text": str(output)}
                
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}") from e
    
    def transcribe_text_only(self, audio_path: str) -> str:
        """
        Transcribe audio and return only the text (convenience method).
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Transcribed text string
        """
        result = self.transcribe(audio_path, timestamps=False)
        return result.get("text", "")
    
    def transcribe_with_timestamps(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio with full timestamp information.
        
        Args:
            audio_path: Path to audio file
        
        Returns:
            Dictionary with text and timestamps at word/segment/char levels
        """
        return self.transcribe(audio_path, timestamps=True)
    
    def transcribe_from_microphone(
        self,
        card: int = 1,
        duration: float = 5.0,
        sample_rate: int = 16000,
        timestamps: bool = False,
        cleanup: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Capture audio from microphone and transcribe in real-time.
        
        Args:
            card: ALSA card number (default: 1 for USB Lavalier Microphone)
            duration: Recording duration in seconds
            sample_rate: Audio sample rate (16kHz required for Parakeet)
            timestamps: Whether to return timestamps
            cleanup: Whether to delete temporary audio file after transcription
        
        Returns:
            Dictionary with transcription and optional timestamps, or None on error
        """
        # Capture audio from microphone
        audio_path = capture_audio_arecord(
            card=card,
            duration=duration,
            sample_rate=sample_rate
        )
        
        if audio_path is None:
            return None
        
        try:
            # Transcribe the captured audio
            result = self.transcribe(audio_path, timestamps=timestamps)
            return result
        finally:
            # Clean up temporary file
            if cleanup:
                try:
                    Path(audio_path).unlink()
                except:
                    pass
    
    def transcribe_microphone_text_only(
        self,
        card: int = 1,
        duration: float = 5.0,
        sample_rate: int = 16000
    ) -> Optional[str]:
        """
        Capture audio from microphone and return transcribed text only.
        
        Args:
            card: ALSA card number (default: 1 for USB Lavalier Microphone)
            duration: Recording duration in seconds
            sample_rate: Audio sample rate (16kHz required for Parakeet)
        
        Returns:
            Transcribed text string, or None on error
        """
        result = self.transcribe_from_microphone(
            card=card,
            duration=duration,
            sample_rate=sample_rate,
            timestamps=False
        )
        
        if result:
            return result.get("text", "").strip()
        return None
    
    def listen_continuous(
        self,
        card: int = 1,
        chunk_duration: float = 5.0,
        sample_rate: int = 16000,
        max_silence_chunks: int = 5,
        silence_chunks_after_speech: int = 2,
        timestamps: bool = False
    ) -> Optional[str]:
        """
        Continuously listen to microphone and return transcription when speech is detected.
        Waits for silence after speech ends before returning (to avoid cutting off sentences).
        
        Args:
            card: ALSA card number (default: 1 for USB Lavalier Microphone)
            chunk_duration: Duration of each audio chunk to check (seconds)
            sample_rate: Audio sample rate (16kHz required for Parakeet)
            max_silence_chunks: Maximum consecutive silent chunks before showing status (while waiting for speech)
            silence_chunks_after_speech: Number of consecutive silent chunks required after speech to finalize transcription
            timestamps: Whether to include timestamps in result
        
        Returns:
            Transcribed text when speech detected and silence follows, or None on error/interrupt
        """
        print(f"[ParakeetSTT] Continuous listening started (Card: {card})")
        print("[ParakeetSTT] Speak your prompt... (Ctrl+C to stop)")
        
        consecutive_silence_count = 0
        accumulated_transcriptions = []
        speech_detected = False
        silence_after_speech_count = 0
        
        while True:
            try:
                # Record a chunk
                result = self.transcribe_from_microphone(
                    card=card,
                    duration=chunk_duration,
                    sample_rate=sample_rate,
                    timestamps=timestamps,
                    cleanup=True
                )
                
                if result:
                    text = result.get("text", "").strip()
                    if text:
                        # Speech detected - accumulate transcriptions
                        speech_detected = True
                        accumulated_transcriptions.append(text)
                        silence_after_speech_count = 0  # Reset silence counter
                        print(f"[ParakeetSTT] Detected: \"{text}\" (listening for more...)")
                    else:
                        # Silence detected
                        if speech_detected:
                            # We've detected speech before, now checking for silence after speech
                            silence_after_speech_count += 1
                            if silence_after_speech_count >= silence_chunks_after_speech:
                                # Enough silence after speech - finalize transcription
                                final_transcription = " ".join(accumulated_transcriptions).strip()
                                print(f"[ParakeetSTT] Final transcription: \"{final_transcription}\"")
                                return final_transcription
                        else:
                            # No speech detected yet - continue listening
                            consecutive_silence_count += 1
                            if consecutive_silence_count <= max_silence_chunks:
                                # Show listening indicator
                                print(".", end="", flush=True)
                            else:
                                # Reset counter and show status
                                consecutive_silence_count = 0
                                print("\n[ParakeetSTT] Still listening... (speak now or Ctrl+C to stop)")
                else:
                    # Error recording - continue trying
                    if speech_detected:
                        silence_after_speech_count += 1
                        if silence_after_speech_count >= silence_chunks_after_speech:
                            # Finalize with what we have
                            final_transcription = " ".join(accumulated_transcriptions).strip()
                            print(f"[ParakeetSTT] Final transcription: \"{final_transcription}\"")
                            return final_transcription
                    else:
                        consecutive_silence_count += 1
                        if consecutive_silence_count <= max_silence_chunks:
                            print(".", end="", flush=True)
                        else:
                            consecutive_silence_count = 0
                            print("\n[ParakeetSTT] Still listening... (speak now or Ctrl+C to stop)")
                        
            except KeyboardInterrupt:
                print("\n[ParakeetSTT] Listening interrupted")
                # Return accumulated transcription if any
                if accumulated_transcriptions:
                    return " ".join(accumulated_transcriptions).strip()
                return None
            except Exception as e:
                print(f"\n[ParakeetSTT] Error during listening: {e}")
                # Return accumulated transcription if any
                if accumulated_transcriptions:
                    return " ".join(accumulated_transcriptions).strip()
                return None


def create_parakeet_stt(model_path: Optional[str] = None, device: Optional[str] = None) -> ParakeetSTT:
    """
    Factory function to create ParakeetSTT instance.
    
    Args:
        model_path: Optional path to .nemo file
        device: Optional device ('cuda', 'cpu', or None for auto-detect)
    
    Returns:
        ParakeetSTT instance
    """
    return ParakeetSTT(model_path=model_path, device=device)


if __name__ == "__main__":
    """Test Parakeet STT with audio file or real-time microphone."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Parakeet STT")
    parser.add_argument(
        "audio_file",
        nargs="?",
        default=None,
        help="Path to audio file (WAV or FLAC, 16kHz mono). If not provided, uses microphone."
    )
    parser.add_argument(
        "--microphone",
        "-m",
        action="store_true",
        help="Use microphone instead of audio file"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to .nemo checkpoint file (default: auto-detect)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use (default: auto-detect - tries CUDA if available, otherwise CPU)"
    )
    parser.add_argument(
        "--card",
        type=int,
        default=1,
        help="ALSA card number for microphone (default: 1 = USB Lavalier)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Recording duration in seconds for microphone (default: 5.0)"
    )
    parser.add_argument(
        "--continuous",
        "-c",
        action="store_true",
        help="Continuous listening mode (keeps listening until speech detected or Ctrl+C)"
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Include timestamps in output"
    )
    
    args = parser.parse_args()
    
    if not NEMO_AVAILABLE:
        print("[ERROR] NeMo toolkit not installed.")
        print("Install with: pip install -U nemo_toolkit[\"asr\"]")
        sys.exit(1)
    
    try:
        # Create STT instance
        print("[ParakeetSTT] Initializing...")
        stt = create_parakeet_stt(model_path=args.model_path, device=args.device)
        
        # Print device info
        if torch is not None:
            print(f"[ParakeetSTT] PyTorch version: {torch.__version__}")
            print(f"[ParakeetSTT] CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"[ParakeetSTT] CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"[ParakeetSTT] Using device: {stt.device}")
        
        # Determine input source
        use_microphone = args.microphone or args.audio_file is None
        
        if use_microphone:
            # Microphone mode
            if args.continuous:
                # Continuous listening mode
                print(f"[ParakeetSTT] Continuous microphone listening (Card: {args.card})")
                text = stt.listen_continuous(
                    card=args.card,
                    chunk_duration=args.duration,
                    timestamps=args.timestamps
                )
                
                if text:
                    print("\n" + "=" * 60)
                    print("TRANSCRIPTION:")
                    print("=" * 60)
                    print(text)
                else:
                    print("\n[ParakeetSTT] No transcription (interrupted or error)")
            else:
                # Single recording
                print(f"[ParakeetSTT] Recording from microphone (Card: {args.card}, Duration: {args.duration}s)...")
                print("  Speak now...")
                
                if args.timestamps:
                    result = stt.transcribe_from_microphone(
                        card=args.card,
                        duration=args.duration,
                        timestamps=True
                    )
                    if result:
                        print("\n" + "=" * 60)
                        print("TRANSCRIPTION:")
                        print("=" * 60)
                        print(result["text"])
                        if "timestamps" in result:
                            print("\n" + "=" * 60)
                            print("TIMESTAMPS:")
                            print("=" * 60)
                            print("\nWord-level timestamps (first 10):")
                            for ts in result["timestamps"].get("word", [])[:10]:
                                print(f"  {ts}")
                            print("\nSegment-level timestamps:")
                            for ts in result["timestamps"].get("segment", []):
                                print(f"  {ts.get('start', 'N/A')}s - {ts.get('end', 'N/A')}s : {ts.get('segment', 'N/A')}")
                else:
                    text = stt.transcribe_microphone_text_only(
                        card=args.card,
                        duration=args.duration
                    )
                    if text:
                        print("\n" + "=" * 60)
                        print("TRANSCRIPTION:")
                        print("=" * 60)
                        print(text)
                    else:
                        print("\n[ParakeetSTT] No transcription (silence or error)")
        else:
            # Audio file mode
            print(f"[ParakeetSTT] Transcribing: {args.audio_file}")
            if args.timestamps:
                result = stt.transcribe_with_timestamps(args.audio_file)
                print("\n" + "=" * 60)
                print("TRANSCRIPTION:")
                print("=" * 60)
                print(result["text"])
                if "timestamps" in result:
                    print("\n" + "=" * 60)
                    print("TIMESTAMPS:")
                    print("=" * 60)
                    print("\nWord-level timestamps (first 10):")
                    for ts in result["timestamps"].get("word", [])[:10]:
                        print(f"  {ts}")
                    print("\nSegment-level timestamps:")
                    for ts in result["timestamps"].get("segment", []):
                        print(f"  {ts.get('start', 'N/A')}s - {ts.get('end', 'N/A')}s : {ts.get('segment', 'N/A')}")
            else:
                text = stt.transcribe_text_only(args.audio_file)
                print("\n" + "=" * 60)
                print("TRANSCRIPTION:")
                print("=" * 60)
                print(text)
        
        print("\n[ParakeetSTT] Done!")
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

