#!/usr/bin/env python3
"""
Test Whisper Speech-to-Text with microphone using arecord (no PyAudio needed).
Captures audio using arecord command and transcribes it using faster-whisper.
"""
import sys
import time
import subprocess
import tempfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("[ERROR] faster-whisper not installed. Install with:")
    print("  pip install faster-whisper>=0.3.0")
    sys.exit(1)


def list_audio_devices():
    """List available audio input devices using arecord."""
    print("\n" + "=" * 60)
    print("Available Audio Input Devices:")
    print("=" * 60)
    
    try:
        result = subprocess.run(
            ['arecord', '-l'],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to list devices: {e}")
        return []
    
    print("=" * 60 + "\n")


def capture_audio_arecord(card=1, device=0, duration=5, sample_rate=16000, channels=1):
    """
    Capture audio using arecord command.
    
    Args:
        card: ALSA card number (e.g., 1 for "USB Lavalier Microphone")
        device: ALSA device number (usually 0)
        duration: Recording duration in seconds
        sample_rate: Sample rate (16kHz recommended for Whisper)
        channels: Number of audio channels (1 = mono)
    
    Returns:
        Path to temporary WAV file
    """
    print(f"\n[RECORDING] Capturing {duration} seconds of audio...")
    print(f"  Card: {card}, Device: {device}")
    print("  Speak now...\n")
    
    # Create temporary WAV file
    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_wav_path = temp_wav.name
    temp_wav.close()
    
    # Build arecord command
    # Format: arecord -D hw:CARD,DEVICE -f S16_LE -r RATE -c CHANNELS -d DURATION output.wav
    # Note: arecord requires integer duration, so convert float to int
    duration_int = int(round(duration))
    cmd = [
        'arecord',
        '-D', f'hw:{card},{device}',
        '-f', 'S16_LE',  # 16-bit signed little-endian
        '-r', str(sample_rate),
        '-c', str(channels),
        '-d', str(duration_int),  # Convert to integer
        temp_wav_path
    ]
    
    try:
        # Run arecord with progress
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Show progress
        elapsed = 0
        while elapsed < duration:
            time.sleep(0.5)
            elapsed += 0.5
            print(f"\r  Recording: {elapsed:.1f}/{duration:.1f}s", end="", flush=True)
        
        print()  # New line after progress
        
        # Wait for process to finish
        stdout, stderr = process.communicate(timeout=2)
        
        if process.returncode != 0:
            print(f"[ERROR] arecord failed: {stderr}")
            return None
        
        print(f"[OK] Audio saved to: {temp_wav_path}")
        return temp_wav_path
        
    except Exception as e:
        print(f"[ERROR] Recording failed: {e}")
        return None


def transcribe_audio(audio_path, model_size="base", device="cpu", compute_type="int8"):
    """
    Transcribe audio using Whisper.
    
    Args:
        audio_path: Path to audio file
        model_size: Whisper model size (tiny, base, small, etc.)
        device: Device (cpu or cuda)
        compute_type: Compute type (int8, float16, float32)
    
    Returns:
        Transcribed text
    """
    print(f"\n[TRANSCRIBING] Loading Whisper model ({model_size}, {device}, {compute_type})...")
    
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("[OK] Model loaded")
        
        print(f"[TRANSCRIBING] Processing audio: {audio_path}")
        segments, info = model.transcribe(audio_path, beam_size=5)
        
        print(f"[OK] Detected language: {info.language} (probability: {info.language_probability:.2f})")
        
        # Collect transcription
        transcription_parts = []
        print("\n[TRANSCRIPTION]")
        print("-" * 60)
        
        for segment in segments:
            text = segment.text.strip()
            if text:
                transcription_parts.append(text)
                print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {text}")
        
        print("-" * 60)
        
        full_text = " ".join(transcription_parts).strip()
        return full_text
        
    except Exception as e:
        print(f"[ERROR] Transcription failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main test function."""
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description="Test Whisper Speech-to-Text with microphone (using arecord)")
    parser.add_argument(
        "--card",
        type=int,
        default=1,
        help="ALSA card number (default: 1 = USB Lavalier Microphone)"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="ALSA device number (default: 0)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Recording duration in seconds (default: 5)"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)"
    )
    parser.add_argument(
        "--device-type",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device type (default: cpu)"
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default="int8",
        choices=["int8", "float16", "float32"],
        help="Compute type (default: int8)"
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Whisper Speech-to-Text Test (using arecord)")
    print("=" * 60)
    
    # List devices if requested
    if args.list_devices:
        list_audio_devices()
        return
    
    # Capture audio
    audio_path = capture_audio_arecord(
        card=args.card,
        device=args.device,
        duration=args.duration,
        sample_rate=16000,
        channels=1
    )
    
    if not audio_path:
        print("[ERROR] Failed to capture audio")
        return
    
    # Transcribe
    try:
        transcription = transcribe_audio(
            audio_path=audio_path,
            model_size=args.model_size,
            device=args.device_type,
            compute_type=args.compute_type
        )
        
        if transcription:
            print(f"\n[RESULT] Full transcription:")
            print(f"  \"{transcription}\"")
        else:
            print("\n[WARNING] No transcription generated (might be silence or error)")
    
    finally:
        # Clean up temp file
        try:
            Path(audio_path).unlink()
        except:
            pass


if __name__ == "__main__":
    main()

