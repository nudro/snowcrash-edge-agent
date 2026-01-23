#!/usr/bin/env python3
"""
Main entry point for Snowcrash.
Starts both the agentic agent and the web-based tracking viewer GUI.
Supports Speech-to-Text (STT) mode using NVIDIA Parakeet (primary) or Whisper (backup).
"""
import sys
import os
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


def kill_gpu_processes():
    """Kill any existing GPU processes to free memory before starting."""
    print("[MAIN] Checking for GPU processes to kill...")
    password = 'baldur123'
    current_pid = os.getpid()
    
    try:
        # Find processes using GPU devices
        gpu_pids = set()
        
        # Method 1: Use fuser to find processes using GPU devices (most reliable)
        try:
            for device in ['/dev/nvidia0', '/dev/nvidiactl', '/dev/nvhost-ctrl', '/dev/nvhost-ctrl-gpu']:
                try:
                    result = subprocess.run(
                        ['fuser', device],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        universal_newlines=True
                    )
                    if result.returncode == 0 and result.stdout:
                        for pid_str in result.stdout.strip().split():
                            try:
                                pid = int(pid_str)
                                if pid != current_pid:
                                    gpu_pids.add(pid)
                            except ValueError:
                                pass
                except FileNotFoundError:
                    pass
        except Exception as e:
            print(f"[MAIN] fuser method failed: {e}")
        
        # Method 2: Check lsof for /dev/nvidia* devices
        try:
            result = subprocess.run(
                ['lsof', '/dev/nvidia0', '/dev/nvidiactl', '/dev/nvhost-ctrl', '/dev/nvhost-ctrl-gpu'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            if result.returncode == 0 and result.stdout:
                for line in result.stdout.split('\n')[1:]:  # Skip header
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                pid = int(parts[1])
                                if pid != current_pid:
                                    gpu_pids.add(pid)
                            except ValueError:
                                pass
        except FileNotFoundError:
            pass  # lsof not available
        
        # Method 3: Check for Python processes that might be using GPU
        try:
            result = subprocess.run(
                ['ps', 'aux'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    line_lower = line.lower()
                    if 'python' in line_lower and any(keyword in line_lower for keyword in ['yolo', 'main', 'tracking', 'tensorrt', 'trt', 'cuda']):
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                pid = int(parts[1])
                                if pid != current_pid:
                                    gpu_pids.add(pid)
                            except ValueError:
                                pass
        except Exception:
            pass
        
        # Method 4: Check for processes using TensorRT/TRT libraries
        try:
            result = subprocess.run(
                ['ps', 'aux'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if any(keyword in line.lower() for keyword in ['tensorrt', 'trt', 'libnvinfer', '.engine']):
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                pid = int(parts[1])
                                if pid != current_pid:
                                    gpu_pids.add(pid)
                            except ValueError:
                                pass
        except Exception:
            pass
        
        if gpu_pids:
            print(f"[MAIN] Found {len(gpu_pids)} GPU processes to kill: {gpu_pids}")
            killed_count = 0
            for pid in gpu_pids:
                try:
                    # First try with sudo (more reliable)
                    kill_cmd = f'echo "{password}" | sudo -S kill -9 {pid} 2>/dev/null'
                    result = subprocess.run(kill_cmd, shell=True, check=False,
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    if result.returncode == 0:
                        killed_count += 1
                        print(f"[MAIN] Killed PID {pid} with sudo")
                    else:
                        # Fallback to normal kill
                        subprocess.run(['kill', '-9', str(pid)], check=False, 
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        killed_count += 1
                        print(f"[MAIN] Killed PID {pid}")
                except Exception as e:
                    print(f"[MAIN] Failed to kill PID {pid}: {e}")
            
            # Wait for processes to die and verify
            time.sleep(2)
            
            # Verify processes are dead
            remaining = []
            for pid in gpu_pids:
                try:
                    # Check if process still exists
                    subprocess.run(['kill', '-0', str(pid)], check=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    remaining.append(pid)
                except subprocess.CalledProcessError:
                    pass  # Process is dead
            
            if remaining:
                print(f"[MAIN] WARNING: {len(remaining)} processes still running: {remaining}")
                # Try one more time with sudo
                for pid in remaining:
                    kill_cmd = f'echo "{password}" | sudo -S kill -9 {pid} 2>/dev/null'
                    subprocess.run(kill_cmd, shell=True, check=False,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                print(f"[MAIN] Successfully killed {killed_count} GPU processes")
        else:
            print("[MAIN] No GPU processes found to kill")
            
        # Clear GPU cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                torch.cuda.empty_cache()  # Clear again
                print("[MAIN] GPU cache cleared")
        except ImportError:
            pass
        except Exception as e:
            print(f"[MAIN] Warning: Could not clear GPU cache: {e}")
            
    except Exception as e:
        print(f"[WARNING] Error killing GPU processes: {e}")
        import traceback
        traceback.print_exc()


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
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        if process.returncode != 0:
            print(f"[ERROR] arecord failed: {process.stderr}")
            return None
        
        return temp_wav_path
    except Exception as e:
        print(f"[ERROR] Failed to capture audio: {e}")
        return None


def transcribe_audio(whisper_model, audio_path, language=None):
    """
    Transcribe audio file using Whisper.
    
    Args:
        whisper_model: WhisperModel instance
        audio_path: Path to WAV file
        language: Optional language code (e.g., 'en')
    
    Returns:
        Transcribed text, or None on error
    """
    if whisper_model is None:
        return None
    
    try:
        # WhisperModel.transcribe() returns a dict with 'text' key
        result = whisper_model.transcribe(audio_path, language=language)
        
        if isinstance(result, dict):
            text = result.get('text', '').strip()
        elif isinstance(result, str):
            text = result.strip()
        else:
            text = str(result).strip()
        
        return text if text else None
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
    Continuously listen for speech and transcribe using STT model.
    
    Args:
        stt_model: ParakeetSTT or WhisperModel instance
        card: ALSA card number
        chunk_duration: Duration of each audio chunk in seconds
        sample_rate: Audio sample rate
        language: Optional language code
        stt_type: "parakeet" or "whisper"
    
    Returns:
        First transcription received, or None if interrupted
    """
    if stt_type == "parakeet" and hasattr(stt_model, 'listen_continuous'):
        # Use Parakeet's built-in continuous listening
        print(f"[STT] Using Parakeet continuous listening (Card: {card})")
        print("[STT] Speak your prompt... (Ctrl+C to stop)")
        try:
            transcription = stt_model.listen_continuous(
                card=card,
                chunk_duration=chunk_duration,
                sample_rate=sample_rate,
                max_silence_chunks=5,
                silence_chunks_after_speech=2,
                timestamps=False
            )
            return transcription.strip() if transcription else None
        except KeyboardInterrupt:
            print("\n[STT] Listening interrupted")
            return None
        except Exception as e:
            print(f"\n[STT] Error during listening: {e}")
            return None
    
    # Fallback to Whisper or manual loop
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
    # Kill GPU processes first to prevent memory issues
    kill_gpu_processes()
    
    parser = argparse.ArgumentParser(description="Snowcrash: Agentic SLM with Tracking Viewer")
    parser.add_argument("--model", choices=["phi-3", "llama", "gemma"], 
                       required=True,
                       help="Model type: phi-3, llama, or gemma (required)")
    parser.add_argument("--temperature", type=float, default=0.7, 
                       help="LLM temperature (default: 0.7)")
    parser.add_argument("--no-gui", action="store_false", dest="gui", default=True,
                       help="Disable web-based tracking viewer GUI (default: GUI enabled)")
    parser.add_argument("--gui-viewer", type=str, choices=["default", "chatgui", "audiogui"], default="chatgui",
                       help="GUI viewer type: 'default' (original), 'chatgui' (with chat), or 'audiogui' (with audio visualization). Default: chatgui")
    parser.add_argument("--gui-port", type=int, default=8083, 
                       help="Port for web viewer GUI (default: 8083)")
    parser.add_argument("--gui-device", type=int, default=0, 
                       help="Camera device for GUI (default: 0)")
    parser.add_argument("--old-nano-host", type=str, 
                       default=os.environ.get("OLD_NANO_HOST", "old-nano"),
                       help="Hostname or IP of old-nano YOLO service (default: old-nano or OLD_NANO_HOST env var)")
    
    # STT (Speech-to-Text) options
    parser.add_argument("--stt", action="store_true",
                       help="Enable Speech-to-Text mode (uses Whisper)")
    parser.add_argument("--stt-card", type=int, default=1,
                       help="Audio card number for STT (default: 1 = USB Lavalier Microphone)")
    parser.add_argument("--stt-chunk-duration", type=float, default=5.0,
                       help="Audio chunk duration in seconds for continuous STT listening (default: 5.0 for better quality)")
    parser.add_argument("--stt-model-size", type=str, default="base",
                       help="Whisper model size for STT (tiny, base, small, medium, large). Default: base")
    parser.add_argument("--stt-model-path", type=str, default=None,
                       help="Path to Parakeet .nemo model file (required for Parakeet STT)")
    parser.add_argument("--stt-type", choices=["whisper", "parakeet"], default="parakeet",
                       help="STT engine: 'whisper' or 'parakeet' (default: parakeet)")
    parser.add_argument("--gui-stt-audio", action="store_true",
                       help="Enable STT audio input in chatgui (requires --stt-model-path for Parakeet)")
    
    args = parser.parse_args()
    
    # Initialize STT model if STT mode is enabled
    parakeet_stt_instance = None
    whisper_model = None
    
    if args.stt:
        if args.stt_type == "parakeet":
            # Parakeet STT (GPU-accelerated, recommended)
            if not args.stt_model_path:
                print("[ERROR] --stt-model-path required for Parakeet STT")
                print("[ERROR] Example: --stt-model-path /path/to/parakeet_rnnt_1.1b_fastpitch.nemo")
                sys.exit(1)
            
            try:
                from tools.parakeet_stt import create_parakeet_stt
                print("[MAIN] Initializing Parakeet STT (GPU)...")
                parakeet_stt_instance = create_parakeet_stt(model_path=args.stt_model_path, device="cuda")
                print("[OK] Parakeet STT initialized")
            except Exception as e:
                print(f"[ERROR] Failed to initialize Parakeet STT: {e}")
                print("[ERROR] Falling back to Whisper STT...")
                args.stt_type = "whisper"
        
        if args.stt_type == "whisper":
            # Whisper STT (CPU, fallback)
            try:
                from faster_whisper import WhisperModel
                print(f"[MAIN] Initializing Whisper STT (model: {args.stt_model_size})...")
                whisper_model = WhisperModel(args.stt_model_size, device="cpu", compute_type="int8")
                print("[OK] Whisper STT initialized")
            except ImportError:
                print("[ERROR] faster-whisper not installed. Install with: pip install faster-whisper")
                print("[ERROR] STT mode disabled")
                args.stt = False
            except Exception as e:
                print(f"[ERROR] Failed to initialize Whisper STT: {e}")
                print("[ERROR] STT mode disabled")
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
                from tools.tracking_viewer_chatgui_V2 import TrackingViewerJanChat
                
                print("[MAIN] Starting Jan.ai chat viewer V2 (simplified)...")
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
    print("=" * 60)
    print()
    
    # Main interaction loop
    try:
        if args.stt:
            # STT mode: continuous listening
            stt_model = parakeet_stt_instance if parakeet_stt_instance else whisper_model
            stt_type = "parakeet" if parakeet_stt_instance else "whisper"
            
            while True:
                try:
                    # Listen for speech
                    transcription = continuous_stt_listening(
                        stt_model=stt_model,
                        card=args.stt_card,
                        chunk_duration=args.stt_chunk_duration,
                        sample_rate=16000,
                        language="en",
                        stt_type=stt_type
                    )
                    
                    if transcription:
                        print(f"\n[USER] {transcription}")
                        
                        # Send to agent
                        if hasattr(agent, 'run_sync'):
                            response = agent.run_sync(transcription)
                        else:
                            import asyncio
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            try:
                                response = loop.run_until_complete(agent.run(transcription))
                            finally:
                                loop.close()
                        
                        print(f"[AGENT] {response}\n")
                    else:
                        print("[STT] No speech detected, continuing to listen...")
                        
                except KeyboardInterrupt:
                    print("\n[MAIN] Stopping STT listening...")
                    break
                except Exception as e:
                    print(f"[ERROR] Error in STT loop: {e}")
                    time.sleep(1)
        else:
            # Text mode: read from stdin
            while True:
                try:
                    prompt = input("You: ").strip()
                    
                    if not prompt:
                        continue
                    
                    if prompt.lower() in ['exit', 'quit']:
                        print("[MAIN] Exiting...")
                        break
                    
                    # Send to agent
                    if hasattr(agent, 'run_sync'):
                        response = agent.run_sync(prompt)
                    else:
                        import asyncio
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            response = loop.run_until_complete(agent.run(prompt))
                        finally:
                            loop.close()
                    
                    print(f"Agent: {response}\n")
                    
                except KeyboardInterrupt:
                    print("\n[MAIN] Exiting...")
                    break
                except EOFError:
                    print("\n[MAIN] Exiting...")
                    break
                except Exception as e:
                    print(f"[ERROR] Error: {e}")
                    import traceback
                    traceback.print_exc()
    
    finally:
        print("\n[MAIN] Shutting down...")
        if viewer_thread:
            print("[MAIN] Web viewer will stop when main process exits")


if __name__ == "__main__":
    main()

