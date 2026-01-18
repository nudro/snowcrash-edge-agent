#!/usr/bin/env python3
"""
Main entry point for Snowcrash.
Starts both the agentic agent and the web-based tracking viewer GUI.
"""
import sys
import argparse
import threading
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.simple_agent import SimpleAgent


def start_web_viewer(port=8080, device=0, duration=0):
    """Start the web-based tracking viewer in a background thread."""
    try:
        from tools.tracking_web_viewer import TrackingWebViewer
        
        print("[MAIN] Starting web-based tracking viewer...")
        viewer = TrackingWebViewer(
            model_path="yolov8n.pt",
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
    """Main entry point - starts agent and optional GUI."""
    parser = argparse.ArgumentParser(description="Snowcrash: Agentic SLM with Tracking Viewer")
    parser.add_argument("--model", choices=["phi-3", "llama", "gemma"], 
                       help="Model type: phi-3, llama, or gemma")
    parser.add_argument("--model-path", type=str, 
                       help="Direct path to GGUF model file (overrides --model)")
    parser.add_argument("--temperature", type=float, default=0.7, 
                       help="LLM temperature (default: 0.7)")
    parser.add_argument("--no-gui", action="store_true", 
                       help="Disable web-based tracking viewer GUI")
    parser.add_argument("--gui-port", type=int, default=8080, 
                       help="Port for web viewer GUI (default: 8080)")
    parser.add_argument("--gui-device", type=int, default=0, 
                       help="Camera device for GUI (default: 0)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Snowcrash - Agentic SLM with Object Tracking")
    print("=" * 60)
    print()
    
    # Initialize agent first (needed for viewer chat)
    print("[MAIN] Initializing agent...")
    agent = SimpleAgent(
        model_path=args.model_path,
        model_type=args.model,
        temperature=args.temperature,
        verbose=True,
        web_viewer=None  # Will be set after viewer is created
    )
    
    # Start web viewer if not disabled
    viewer_instance = None
    viewer_thread = None
    if not args.no_gui:
        try:
            from tools.tracking_web_viewer import TrackingWebViewer
            
            print("[MAIN] Starting web-based tracking viewer...")
            viewer_instance = TrackingWebViewer(
                model_path="yolov8n.pt",
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
            print(f"[WARNING] Failed to start web viewer: {e}")
            viewer_instance = None
    
    print()
    print("=" * 60)
    print("Agent ready! Type your prompts below.")
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
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 60)
    print()
    
    # Interactive loop
    try:
        while True:
            try:
                prompt = input("You: ").strip()
                
                if not prompt:
                    continue
                
                if prompt.lower() in ["exit", "quit", "q"]:
                    print("\nShutting down...")
                    break
                
                # Run agent
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

