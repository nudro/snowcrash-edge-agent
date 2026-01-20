#!/usr/bin/env python3
"""
Main entry point for Snowcrash.
Starts both the agentic agent and the web-based tracking viewer GUI.
Supports Speech-to-Text (STT) mode using NVIDIA Parakeet (primary) or Whisper (backup).
"""
import sys
import argparse
import threading
import time
import subprocess
import tempfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.simple_agent import SimpleAgent


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


def transcribe_audio(whisper_model, audio_path, language=None):
    """
    Transcribe audio using Whisper model with improved parameters.
    
    Args:
        whisper_model: WhisperModel instance
        audio_path: Path to audio file
        language: Optional language code (e.g., "en", "es") for better accuracy
    
    Returns:
        Transcribed text, or None on error
    """
    try:
        # Improved transcription parameters for better quality:
        # - beam_size=5: Good balance between speed and quality
        # - language: Specify language for better accuracy (or None for auto-detect)
        transcribe_kwargs = {
            "beam_size": 5,
            "language": language  # Auto-detect if None, or specify for better accuracy
        }
        
        # Optional parameters (may not be available in all faster-whisper versions)
        try:
            # Try to use VAD filter and context if available
            segments, info = whisper_model.transcribe(
                audio_path,
                **transcribe_kwargs,
                vad_filter=True,  # Voice Activity Detection - filters out non-speech
                condition_on_previous_text=True  # Uses previous text for context
            )
        except TypeError:
            # Fallback if parameters not supported
            segments, info = whisper_model.transcribe(audio_path, **transcribe_kwargs)
        
        # Collect transcription
        transcription_parts = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                transcription_parts.append(text)
        
        full_text = " ".join(transcription_parts).strip()
        return full_text if full_text else None
        
    except Exception as e:
        print(f"[ERROR] Transcription failed: {e}")
        return None


def capture_and_transcribe(whisper_model, card=1, duration=3, sample_rate=16000, language=None):
    """
    Capture audio chunk from microphone and transcribe using Whisper.
    
    Args:
        whisper_model: WhisperModel instance
        card: ALSA card number
        duration: Recording duration in seconds (for chunk)
        sample_rate: Audio sample rate
        language: Optional language code for better transcription accuracy
    
    Returns:
        Transcribed text, or None on error/silence
    """
    # Capture audio
    audio_path = capture_audio_arecord(
        card=card,
        device=0,
        duration=duration,
        sample_rate=sample_rate,
        channels=1
    )
    
    if not audio_path:
        return None
    
    try:
        # Transcribe with language hint for better accuracy
        transcription = transcribe_audio(whisper_model, audio_path, language=language)
        return transcription
        
    finally:
        # Clean up temp file
        try:
            Path(audio_path).unlink()
        except:
            pass


def continuous_stt_listening(stt_model, card=1, chunk_duration=3.0, sample_rate=16000, language=None, stt_type="parakeet"):
    """
    Continuously listen for speech and return transcription when speech is detected.
    Keeps listening until speech is detected or interrupted.
    
    Args:
        stt_model: ParakeetSTT instance or WhisperModel instance
        card: ALSA card number
        chunk_duration: Duration of each audio chunk to check (seconds)
        sample_rate: Audio sample rate (16kHz for Parakeet, 16kHz recommended for Whisper)
        language: Optional language code (for Whisper)
        stt_type: "parakeet" or "whisper"
    
    Returns:
        Transcribed text when speech is detected, or None on error/interrupt
    """
    stt_name = "Parakeet" if stt_type == "parakeet" else "Whisper"
    print(f"[STT] Continuous listening started ({stt_name}, Card: {card})")
    print("[STT] Speak your prompt... (Ctrl+C to stop)")
    
    consecutive_silence_count = 0
    max_silence_chunks = 5  # Keep listening even if some chunks are silent
    
    while True:
        try:
            # Record a chunk
            transcription = capture_and_transcribe(
                stt_model=stt_model,
                card=card,
                duration=chunk_duration,
                sample_rate=sample_rate,
                language=language,
                stt_type=stt_type
            )
            
            if transcription and transcription.strip():
                # Speech detected
                print(f"[STT] Detected: \"{transcription}\"")
                return transcription.strip()
            else:
                # Silence - continue listening
                consecutive_silence_count += 1
                if consecutive_silence_count <= max_silence_chunks:
                    # Show listening indicator
                    print(".", end="", flush=True)
                else:
                    # Reset counter and show status
                    consecutive_silence_count = 0
                    print("\n[STT] Still listening... (speak now or Ctrl+C to stop)")
                
        except KeyboardInterrupt:
            print("\n[STT] Listening interrupted")
            return None
        except Exception as e:
            print(f"\n[STT] Error during listening: {e}")
            return None


def start_web_viewer(port=8080, device=0, duration=0):
    """Start the web-based tracking viewer in a background thread."""
    try:
        from tools.tracking_web_viewer import TrackingWebViewer
        
        print("[MAIN] Starting web-based tracking viewer...")
        viewer = TrackingWebViewer(
            model_path="yolo26n-seg.pt",
            device=device,
            confidence_threshold=0.50,
            use_gstreamer=True,
            port=port
        )
        
        # Run viewer in background thread
        def run_viewer():
            viewer.run(duration_seconds=duration)
        
        viewer_thread = threading.Thread(target=run_viewer, daemon=True)
        viewer_thread.start()
        
        # Give it a moment to start
        time.sleep(2)
        
        # Get device IP for display
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            device_ip = s.getsockname()[0]
            s.close()
        except:
            device_ip = "localhost"
        
        print(f"[MAIN] Web viewer started at http://{device_ip}:{port}")
        print(f"[MAIN] Open your browser to view tracking GUI")
        
        return viewer_thread
    except ImportError as e:
        print(f"[WARNING] Flask not available. Web viewer disabled: {e}")
        print("[MAIN] Install Flask with: pip install flask")
        return None
    except Exception as e:
        print(f"[WARNING] Failed to start web viewer: {e}")
        return None


def main():
    """Main entry point - starts agent and GUI (GUI is always enabled)."""
    parser = argparse.ArgumentParser(description="Snowcrash: Agentic SLM with Tracking Viewer")
    parser.add_argument("--model", choices=["phi-3", "llama", "gemma"], 
                       required=True,
                       help="Model type: phi-3, llama, or gemma (required)")
    parser.add_argument("--temperature", type=float, default=0.7, 
                       help="LLM temperature (default: 0.7)")
    parser.add_argument("--no-gui", action="store_false", dest="gui", default=True,
                       help="Disable web-based tracking viewer GUI (default: GUI enabled)")
    parser.add_argument("--gui-viewer", type=str, choices=["default", "chatgui", "audiogui"], default="default",
                       help="GUI viewer type: 'default' (original), 'chatgui' (with chat), or 'audiogui' (with audio visualization). Default: default")
    parser.add_argument("--gui-port", type=int, default=8080, 
                       help="Port for web viewer GUI (default: 8080)")
    parser.add_argument("--gui-device", type=int, default=0, 
                       help="Camera device for GUI (default: 0)")
    
    # STT (Speech-to-Text) options
    parser.add_argument("--stt", action="store_true",
                       help="Enable Speech-to-Text mode (uses Whisper)")
    parser.add_argument("--stt-card", type=int, default=1,
                       help="Audio card number for STT (default: 1 = USB Lavalier Microphone)")
    parser.add_argument("--stt-chunk-duration", type=float, default=5.0,
                       help="Audio chunk duration in seconds for continuous STT listening (default: 5.0 for better quality)")
    parser.add_argument("--stt-model-size", type=str, default="base",
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size for STT (default: base)")
    parser.add_argument("--stt-device-type", type=str, default="cpu",
                       choices=["cpu", "cuda"],
                       help="Device type for Whisper (default: cpu)")
    parser.add_argument("--stt-compute-type", type=str, default="int8",
                       choices=["int8", "float16", "float32"],
                       help="Compute type for Whisper (default: int8)")
    parser.add_argument("--stt-language", type=str, default=None,
                       help="Language code for STT (e.g., 'en', 'es'). Auto-detect if not specified")
    parser.add_argument("--stt-model-path", type=str, default=None,
                       help="Path to Parakeet .nemo model file (for chatgui audio input)")
    parser.add_argument("--gui-stt-audio", action="store_true",
                       help="Enable STT audio input in chatgui (requires --gui-viewer chatgui and Parakeet model)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Snowcrash - Agentic SLM with Object Tracking")
    if args.stt:
        print("  Mode: Speech-to-Text (STT) enabled")
    print("=" * 60)
    print()
    
    # Initialize STT instances (will be populated if needed)
    parakeet_stt_instance = None
    whisper_model = None
    parakeet_stt_for_terminal = None
    
    # Lazy load Parakeet STT for chatgui audio input (if requested)
    if args.gui_stt_audio and args.gui_viewer == "chatgui":
        try:
            from tools.parakeet_stt import create_parakeet_stt
            print("[MAIN] Initializing Parakeet STT for chatgui audio input (GPU)...")
            parakeet_stt_instance = create_parakeet_stt(model_path=args.stt_model_path, device="cuda")
            print("[OK] Parakeet STT initialized for chatgui (GPU)")
        except ImportError:
            print("[ERROR] NVIDIA NeMo toolkit not installed. Install with:")
            print("  pip install -U nemo_toolkit[\"asr\"]")
            print("[ERROR] Chatgui STT audio input disabled.")
            args.gui_stt_audio = False
        except Exception as e:
            print(f"[ERROR] Failed to initialize Parakeet STT: {e}")
            print("[ERROR] Chatgui STT audio input disabled.")
            args.gui_stt_audio = False
    
    # Lazy load Whisper model if STT is enabled (for terminal mode)
    if args.stt:
        try:
            from faster_whisper import WhisperModel
            print(f"[MAIN] Loading Whisper model ({args.stt_model_size}, {args.stt_device_type}, {args.stt_compute_type})...")
            whisper_model = WhisperModel(
                args.stt_model_size,
                device=args.stt_device_type,
                compute_type=args.stt_compute_type
            )
            print("[OK] Whisper model loaded")
            print(f"  Memory usage: ~150-500 MB (model: {args.stt_model_size})")
        except ImportError:
            print("[ERROR] faster-whisper not installed. Install with:")
            print("  pip install faster-whisper>=0.3.0")
            print("[ERROR] STT mode disabled. Falling back to text input.")
            args.stt = False
        except Exception as e:
            print(f"[ERROR] Failed to load Whisper model: {e}")
            print("[ERROR] STT mode disabled. Falling back to text input.")
            args.stt = False
    
    # Initialize agent first (needed for viewer)
    print("[MAIN] Initializing agent...")
    agent = SimpleAgent(
        model_path=None,  # Use model_type instead
        model_type=args.model,
        temperature=args.temperature,
        verbose=True,
        web_viewer=None  # Will be set after viewer is created
    )
    
    # Start web viewer (GUI enabled by default, can be disabled with --no-gui)
    viewer_instance = None
    viewer_thread = None
    if args.gui:
        try:
            # Choose viewer: audiogui, chatgui, or default web_viewer
            if args.gui_viewer == "audiogui":
                from tools.tracking_viewer_audiogui import TrackingViewerAudioGUI
                
                # AudioGUI REQUIRES STT - initialize if not already done
                if parakeet_stt_instance is None:
                    try:
                        from tools.parakeet_stt import create_parakeet_stt
                        print("[MAIN] Initializing Parakeet STT for AudioGUI (required, GPU)...")
                        parakeet_stt_instance = create_parakeet_stt(model_path=args.stt_model_path, device="cuda")
                        print("[OK] Parakeet STT initialized for AudioGUI (GPU)")
                    except Exception as e:
                        print(f"[ERROR] Failed to initialize STT for AudioGUI: {e}")
                        print("[ERROR] AudioGUI requires STT. Exiting.")
                        raise RuntimeError("AudioGUI requires Parakeet STT")
                
                print("[MAIN] Starting AudioGUI viewer (with audio visualization)...")
                viewer_instance = TrackingViewerAudioGUI(
                    model_path="yolo26n-seg.pt",
                    device=args.gui_device,
                    confidence_threshold=0.50,
                    use_gstreamer=True,
                    port=args.gui_port,
                    agent=agent,  # Pass agent instance for chat (shares LLM - no extra memory)
                    stt_model=parakeet_stt_instance,  # REQUIRED for AudioGUI
                    stt_card=args.stt_card,  # USB microphone card number
                    stt_chunk_duration=args.stt_chunk_duration  # Audio chunk duration
                )
            elif args.gui_viewer == "chatgui":
                from tools.tracking_viewer_chatgui import TrackingViewerJanChat
                
                print("[MAIN] Starting Jan.ai chat viewer (experimental)...")
                # Use Parakeet STT instance if available, otherwise None
                stt_for_chatgui = parakeet_stt_instance if args.gui_stt_audio else None
                viewer_instance = TrackingViewerJanChat(
                    model_path="yolo26n-seg.pt",
                    device=args.gui_device,
                    confidence_threshold=0.50,
                    use_gstreamer=True,
                    port=args.gui_port,
                    agent=agent,  # Pass agent instance for chat (shares LLM - no extra memory)
                    stt_model=stt_for_chatgui,  # Pass Parakeet STT instance if audio input enabled
                    stt_enabled=args.gui_stt_audio,  # Enable STT audio input in chatgui
                    stt_card=args.stt_card,  # USB microphone card number
                    stt_chunk_duration=args.stt_chunk_duration  # Audio chunk duration
                )
            else:
                from tools.tracking_web_viewer import TrackingWebViewer
                
                print("[MAIN] Starting web-based tracking viewer...")
                viewer_instance = TrackingWebViewer(
                    model_path="yolo26n-seg.pt",
                    device=args.gui_device,
                    confidence_threshold=0.50,
                    use_gstreamer=True,
                    port=args.gui_port
                )
            
            # Update agent's web_viewer reference
            agent.web_viewer = viewer_instance
            
            # Run viewer in background thread
            def run_viewer():
                viewer_instance.run(duration_seconds=0)  # Run until stopped
            
            viewer_thread = threading.Thread(target=run_viewer, daemon=True)
            viewer_thread.start()
            
            # Give it a moment to start
            time.sleep(2)
            
            # Get device IP for display
            try:
                import socket
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                device_ip = s.getsockname()[0]
                s.close()
            except:
                device_ip = "localhost"
            
            print(f"[MAIN] Web viewer started at http://{device_ip}:{args.gui_port}")
            print(f"[MAIN] Open your browser to view tracking GUI")
        except Exception as e:
            print(f"[ERROR] Failed to start web viewer: {e}")
            print("[ERROR] GUI failed to start. Please check your dependencies and try again.")
            sys.exit(1)
    else:
        print("[MAIN] GUI disabled (--no-gui flag)")
    
    print()
    print("=" * 60)
    if args.stt:
        print("Agent ready! Continuous STT listening enabled.")
        print(f"  Listening continuously (chunk size: {args.stt_chunk_duration}s)")
        print("  Speak your prompts - will transcribe and send to agent automatically.")
        print("  Press Ctrl+C to stop.")
    else:
        print("Agent ready! Type your prompts below.")
        print("Type 'exit' or 'quit' to stop.")
    if viewer_instance:
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            device_ip = s.getsockname()[0]
            s.close()
            print(f"Web viewer: http://{device_ip}:{args.gui_port}")
        except:
            print(f"Web viewer: http://localhost:{args.gui_port}")
    print("=" * 60)
    print()
    
    # Interactive loop
    try:
        while True:
            try:
                # Get prompt from STT or text input
                if args.stt:
                    # STT mode: continuously listen until speech is detected
                    # Use Parakeet if available, otherwise Whisper
                    stt_model = parakeet_stt_for_terminal if parakeet_stt_for_terminal else whisper_model
                    current_stt_type = "parakeet" if parakeet_stt_for_terminal else "whisper"
                    
                    prompt = continuous_stt_listening(
                        stt_model=stt_model,
                        card=args.stt_card,
                        chunk_duration=args.stt_chunk_duration,
                        sample_rate=16000,
                        language=args.stt_language,
                        stt_type=current_stt_type
                    )
                    
                    # Handle interruptions (Ctrl+C or errors)
                    if prompt is None:
                        print("\n[STT] Stopping...")
                        break
                    
                    # Display transcription as if user typed it
                    print(f"\nYou: {prompt}")
                    
                    # Allow exit commands even in STT mode (user could type them)
                    # But in STT mode, we'll use a special phrase or Ctrl+C
                else:
                    # Normal text input mode
                    prompt = input("You: ").strip()
                    
                    if not prompt:
                        continue
                    
                    if prompt.lower() in ["exit", "quit", "q"]:
                        print("\nShutting down...")
                        break
                
                # Run agent (same for both STT and text modes)
                response = agent.run_sync(prompt)
                print(f"Agent: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\nShutting down...")
                break
            except EOFError:
                print("\nShutting down...")
                break
            except Exception as e:
                print(f"[ERROR] {e}\n")
    
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    
    print("[MAIN] Shutdown complete.")


if __name__ == "__main__":
    main()

