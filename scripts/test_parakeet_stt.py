#!/usr/bin/env python3
"""
Test NVIDIA Parakeet STT with microphone.

Captures audio using arecord and transcribes it using Parakeet TDT 0.6B V2 model.
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.parakeet_stt import create_parakeet_stt
from main import capture_audio_arecord


def main():
    parser = argparse.ArgumentParser(description="Test Parakeet STT with microphone")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to .nemo checkpoint file (default: auto-detect)"
    )
    parser.add_argument(
        "--card",
        type=int,
        default=1,
        help="ALSA card number (default: 1)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Recording duration in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate in Hz (default: 16000, required for Parakeet)"
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        help="Include timestamps in output"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Parakeet STT Test")
    print("=" * 60)
    print(f"Card: {args.card}")
    print(f"Duration: {args.duration}s")
    print(f"Sample Rate: {args.sample_rate}Hz")
    print("=" * 60 + "\n")
    
    try:
        # Create STT instance
        print("[ParakeetSTT] Initializing...")
        stt = create_parakeet_stt(model_path=args.model_path)
        print("[OK] Parakeet STT ready\n")
        
        # Capture audio
        print(f"[Recording] Capturing {args.duration}s of audio...")
        print("  Speak now...")
        audio_path = capture_audio_arecord(
            card=args.card,
            duration=args.duration,
            sample_rate=args.sample_rate
        )
        
        if audio_path is None:
            print("[ERROR] Failed to capture audio")
            return 1
        
        print("[OK] Audio captured\n")
        
        # Transcribe
        print("[Transcribing] Processing audio...")
        if args.timestamps:
            result = stt.transcribe_with_timestamps(audio_path)
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
            text = stt.transcribe_text_only(audio_path)
            print("\n" + "=" * 60)
            print("TRANSCRIPTION:")
            print("=" * 60)
            print(text)
        
        # Clean up
        try:
            Path(audio_path).unlink()
        except:
            pass
        
        print("\n" + "=" * 60)
        print("[OK] Test complete!")
        print("=" * 60)
        
        return 0
        
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("\nInstall NeMo toolkit with:")
        print("  pip install -U nemo_toolkit[\"asr\"]")
        return 1
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

