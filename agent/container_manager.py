"""
Container Manager - Manages Docker containers for llama-server.

Handles starting, stopping, and health checking of llama-server containers.
"""
import subprocess
import time
import httpx
from typing import Optional, Dict
from pathlib import Path


class ContainerManager:
    """
    Manages Docker containers running llama-server for different models.
    
    Each model gets its own container on a different port:
    - llama: port 8080
    - phi-3: port 8081
    - gemma: port 8082
    """
    
    # Model to port mapping
    MODEL_PORTS = {
        "llama": 8080,
        "phi-3": 8081,
        "gemma": 8082,
    }
    
    # Model to container name mapping
    MODEL_CONTAINERS = {
        "llama": "snowcrash-llama-server",
        "phi-3": "snowcrash-phi3-server",
        "gemma": "snowcrash-gemma-server",
    }
    
    # Model to model path mapping (inside container)
    MODEL_PATHS = {
        "llama": "/snowcrash/models/llama3.2/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "phi-3": "/snowcrash/models/phi3-mini/Phi-3-mini-4k-instruct-q4.gguf",
        "gemma": "/snowcrash/models/gemma2b/gemma-3n-E2B-it-Q4_K_M.gguf",
    }
    
    def __init__(self, models_base_path: str = "/home/ordun/Documents/snowcrash/models", verbose: bool = True):
        """
        Initialize container manager.
        
        Args:
            models_base_path: Base path to models directory on host
            verbose: Print status messages
        """
        self.models_base_path = Path(models_base_path)
        self.verbose = verbose
        self.running_containers: Dict[str, str] = {}  # model_type -> container_id (only containers we started)
    
    def _run_command(self, cmd: list, check: bool = True) -> tuple[str, str, int]:
        """
        Run shell command and return stdout, stderr, returncode.
        
        Args:
            cmd: Command as list of strings
            check: Raise exception if return code != 0
            
        Returns:
            (stdout, stderr, returncode)
        """
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=check,
                timeout=30  # 30 second timeout
            )
            return result.stdout, result.stderr, result.returncode
        except subprocess.CalledProcessError as e:
            return e.stdout, e.stderr, e.returncode
        except FileNotFoundError:
            # Command not found (e.g., docker not installed)
            return "", "Command not found", 127
        except subprocess.TimeoutExpired:
            return "", "Command timed out", 124
    
    def is_container_running(self, model_type: str) -> bool:
        """
        Check if container for model is already running.
        
        Args:
            model_type: Model type ("llama", "phi-3", "gemma")
            
        Returns:
            True if container is running
        """
        container_name = self.MODEL_CONTAINERS.get(model_type)
        if not container_name:
            return False
        
        stdout, _, returncode = self._run_command(
            ["docker", "ps", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
            check=False
        )
        
        return container_name in stdout
    
    def _wait_for_server_ready(self, port: int, max_wait: int = 60) -> bool:
        """
        Wait for llama-server to be ready on given port.
        
        Args:
            port: Port to check
            max_wait: Maximum seconds to wait (increased to 60 for model loading)
            
        Returns:
            True if server is ready, False if timeout
        """
        start_time = time.time()
        check_interval = 2  # Check every 2 seconds
        
        if self.verbose:
            print(f"[ContainerManager] Checking if llama-server is ready on port {port}...")
        
        while time.time() - start_time < max_wait:
            elapsed = int(time.time() - start_time)
            
            # Try the actual API endpoint (more reliable than /health)
            try:
                response = httpx.post(
                    f"http://localhost:{port}/v1/chat/completions",
                    json={"model": "test", "messages": [{"role": "user", "content": "test"}]},
                    timeout=3
                )
                # If we get any response (even error), server is up
                if self.verbose:
                    print(f"[ContainerManager] llama-server responded on port {port} (elapsed: {elapsed}s)")
                return True
            except httpx.ConnectError:
                # Server not ready yet
                if self.verbose and elapsed % 10 == 0:  # Print every 10 seconds
                    print(f"[ContainerManager] Waiting for llama-server... ({elapsed}s/{max_wait}s)")
            except httpx.TimeoutException:
                # Server might be loading model
                if self.verbose and elapsed % 10 == 0:
                    print(f"[ContainerManager] Server responding slowly, may be loading model... ({elapsed}s/{max_wait}s)")
            except Exception as e:
                # Other error - might be server error, but server is up
                if self.verbose:
                    print(f"[ContainerManager] Server responded with error (server is up): {e}")
                return True
            
            time.sleep(check_interval)
        
        if self.verbose:
            print(f"[ContainerManager] Timeout waiting for llama-server on port {port} after {max_wait}s")
        return False
    
    def start_model_container(self, model_type: str) -> Optional[int]:
        """
        Start Docker container for given model.
        
        Args:
            model_type: Model type ("llama", "phi-3", "gemma")
            
        Returns:
            Port number if successful, None if failed
        """
        if model_type not in self.MODEL_PORTS:
            if self.verbose:
                print(f"[ContainerManager] Unknown model type: {model_type}")
            return None
        
        # Check if already running
        if self.is_container_running(model_type):
            port = self.MODEL_PORTS[model_type]
            if self.verbose:
                print(f"[ContainerManager] Container for {model_type} already running on port {port}")
            return port
        
        port = self.MODEL_PORTS[model_type]
        container_name = self.MODEL_CONTAINERS[model_type]
        model_path = self.MODEL_PATHS[model_type]
        
        # Clear caches to reduce memory fragmentation (like llama_wrapper.sh)
        # This helps prevent CUDA out of memory errors on Jetson
        if self.verbose:
            print(f"[ContainerManager] Clearing system caches to reduce memory fragmentation...")
        try:
            # Sync filesystem
            sync_result = self._run_command(["sync"], check=False)
            # Drop caches (requires sudo, but non-blocking if it fails)
            drop_caches_result = self._run_command(
                ["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
                check=False
            )
            if drop_caches_result[2] == 0:
                if self.verbose:
                    print(f"[ContainerManager] ✓ Caches cleared")
            else:
                if self.verbose:
                    print(f"[ContainerManager] ⚠ Could not clear caches (may require sudo)")
        except Exception as e:
            if self.verbose:
                print(f"[ContainerManager] ⚠ Cache clearing failed (non-critical): {e}")
        
        # Verify model file exists on host before starting container
        # model_path is container path (e.g., /snowcrash/models/llama3.2/file.gguf)
        # Convert to host path by replacing /snowcrash/models with models_base_path
        host_model_path = self.models_base_path / model_path.replace("/snowcrash/models/", "")
        if not host_model_path.exists():
            if self.verbose:
                print(f"[ContainerManager] ❌ Model file not found on host: {host_model_path}")
                print(f"[ContainerManager] Expected location: {host_model_path}")
                print(f"[ContainerManager] Container path: {model_path}")
                print(f"[ContainerManager] Models base path: {self.models_base_path}")
            return None
        
        if self.verbose:
            print(f"[ContainerManager] Starting container for {model_type} on port {port}...")
            print(f"[ContainerManager] Model file: {host_model_path} (exists)")
            print(f"[ContainerManager] Container will use: {model_path}")
        
        # Build docker run command
        # Uses jetson-containers autotag to get the right image
        cmd = [
            "docker", "run",
            "-d",  # Detached mode
            "--name", container_name,
            "--runtime", "nvidia",
            "--env", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
            "--network", "host",  # Use host network for localhost access
            "--shm-size", "2g",  # Shared memory
            # Note: Not setting --memory limit to avoid unified memory fragmentation on Jetson
            # Jetson uses unified memory (CPU+GPU share RAM), so limiting container RAM
            # can cause GPU memory allocation failures even when total memory is sufficient
            "-v", f"{self.models_base_path}:/snowcrash/models:ro",  # Mount models (read-only)
            "--rm",  # Auto-remove on stop
        ]
        
        # Get container image using jetson-containers autotag
        # This gets the correct image tag for the current Jetson setup
        image = None
        try:
            # Try to get image tag using autotag (from jetson-containers)
            # autotag is usually in /usr/local/bin/autotag or installed via jetson-containers
            autotag_cmd = ["autotag", "llama_cpp"]
            stdout, stderr, returncode = self._run_command(autotag_cmd, check=False)
            if returncode == 0 and stdout.strip():
                image = stdout.strip()
                if self.verbose:
                    print(f"[ContainerManager] Using image from autotag: {image}")
            else:
                if self.verbose:
                    # autotag not found is OK - we'll use fallback
                    if returncode == 127:
                        print(f"[ContainerManager] autotag not found (command not found) - using fallback")
                    else:
                        print(f"[ContainerManager] autotag failed (returncode: {returncode}, stderr: {stderr}) - using fallback")
        except Exception as e:
            if self.verbose:
                print(f"[ContainerManager] autotag command failed: {e} - using fallback")
        
        # If autotag failed, try to find a valid image tag
        if not image:
            if self.verbose:
                print("[ContainerManager] autotag failed, checking for existing llama_cpp images...")
            try:
                # Check if any llama_cpp images exist locally
                check_cmd = ["docker", "images", "dustynv/llama_cpp", "--format", "{{.Repository}}:{{.Tag}}"]
                stdout, _, returncode = self._run_command(check_cmd, check=False)
                if returncode == 0 and stdout.strip():
                    # Use the first available image
                    images = stdout.strip().split('\n')
                    image = images[0]
                    if self.verbose:
                        print(f"[ContainerManager] Using existing local image: {image}")
                else:
                    # No local images - raise error with helpful message
                    raise RuntimeError(
                        "autotag failed and no local llama_cpp images found.\n"
                        "Please run on Jetson:\n"
                        "  jetson-containers run $(autotag llama_cpp) echo 'test'\n"
                        "This will pull the correct image. Or manually:\n"
                        "  docker pull dustynv/llama_cpp:r36.4.0  # (adjust version as needed)"
                    )
            except Exception as e:
                if "RuntimeError" in str(type(e)):
                    raise  # Re-raise our helpful error
                raise RuntimeError(
                    f"Failed to get Docker image: {e}\n"
                    "Please ensure jetson-containers is installed and autotag works:\n"
                    "  which autotag\n"
                    "  autotag llama_cpp"
                )
        
        # Add image and llama-server command
        # Use full GPU (-1 = all layers) for maximum performance when YOLO is not running
        # If YOLO is also running, reduce to 6-8 layers to avoid OOM
        cmd.extend([
            image,
            "llama-server",
            "-m", model_path,
            "--port", str(port),
            "--n-gpu-layers", "-1",  # -1 = all layers on GPU (full GPU acceleration)
            "--ctx-size", "2048",  # Increased from 512 to match llama-cpp-python and accommodate prompt templates
            "--batch-size", "128",
        ])
        
        # Run docker command
        stdout, stderr, returncode = self._run_command(cmd, check=False)
        
        if returncode != 0:
            if self.verbose:
                print(f"[ContainerManager] Failed to start container: {stderr}")
            return None
        
        container_id = stdout.strip()
        if not container_id:
            if self.verbose:
                print(f"[ContainerManager] ❌ Failed to start container - no container ID returned")
                print(f"[ContainerManager] stdout: {stdout}")
                print(f"[ContainerManager] stderr: {stderr}")
            return None
        
        self.running_containers[model_type] = container_id
        
        if self.verbose:
            print(f"[ContainerManager] Container started: {container_id[:12]}...")
        
        # Check container status immediately and periodically
        # Container might crash immediately with --rm, so we need to check quickly
        max_checks = 5
        check_interval = 1  # Check every second
        container_running = False
        
        for i in range(max_checks):
            time.sleep(check_interval)
            if self.is_container_running(model_type):
                container_running = True
                break
            # Try to get logs immediately (container might still exist even if not running)
            if self.verbose and i == 0:
                print(f"[ContainerManager] Checking container status...")
        
        if not container_running:
            if self.verbose:
                print(f"[ContainerManager] ❌ Container {container_name} is not running!")
                print(f"[ContainerManager] Container may have crashed immediately.")
                print(f"[ContainerManager] Attempting to get logs (container may have been removed)...")
                
                # Try to get logs - container might still exist briefly
                try:
                    logs_cmd = ["docker", "logs", "--tail", "50", container_name]
                    logs_stdout, logs_stderr, logs_rc = self._run_command(logs_cmd, check=False)
                    if logs_rc == 0:
                        if logs_stdout:
                            print(f"[ContainerManager] Container logs:")
                            print(logs_stdout)
                        if logs_stderr:
                            print(f"[ContainerManager] Container errors:")
                            print(logs_stderr)
                    else:
                        print(f"[ContainerManager] Could not get logs - container already removed")
                        print(f"[ContainerManager] This usually means the container crashed immediately")
                        print(f"[ContainerManager] Possible causes:")
                        print(f"[ContainerManager]   1. Model file not found: {model_path}")
                        print(f"[ContainerManager]   2. GPU memory insufficient")
                        print(f"[ContainerManager]   3. Invalid model file")
                        print(f"[ContainerManager]   4. Docker image issue")
                        print(f"[ContainerManager]")
                        print(f"[ContainerManager] Try running manually to see error:")
                        print(f"[ContainerManager]   docker run --rm --runtime nvidia --network host \\")
                        print(f"[ContainerManager]     -v {self.models_base_path}:/models:ro \\")
                        print(f"[ContainerManager]     $(autotag llama_cpp) \\")
                        print(f"[ContainerManager]     llama-server -m {model_path} --port {port} --n-gpu-layers 8")
                except Exception as e:
                    print(f"[ContainerManager] Could not get container logs: {e}")
            return None
        
        if self.verbose:
            print(f"[ContainerManager] Container is running, waiting for llama-server to be ready...")
        
        # Wait for server to be ready, but check container status periodically
        # If container crashes, we want to know immediately
        if self.verbose:
            print(f"[ContainerManager] Container is running, waiting for llama-server to be ready...")
        
        # Monitor container while waiting for server
        server_ready = False
        start_time = time.time()
        max_wait = 60
        check_interval = 2
        
        while time.time() - start_time < max_wait:
            # Check if container is still running
            if not self.is_container_running(model_type):
                if self.verbose:
                    print(f"[ContainerManager] ❌ Container crashed while waiting for server!")
                    print(f"[ContainerManager] Attempting to get logs...")
                    try:
                        logs_cmd = ["docker", "logs", "--tail", "50", container_name]
                        logs_stdout, logs_stderr, _ = self._run_command(logs_cmd, check=False)
                        if logs_stdout:
                            print(f"[ContainerManager] Container logs:")
                            print(logs_stdout)
                        if logs_stderr:
                            print(f"[ContainerManager] Container errors:")
                            print(logs_stderr)
                    except Exception as e:
                        print(f"[ContainerManager] Could not get logs: {e}")
                return None
            
            # Check if server is ready
            try:
                response = httpx.post(
                    f"http://localhost:{port}/v1/chat/completions",
                    json={"model": "test", "messages": [{"role": "user", "content": "test"}]},
                    timeout=2
                )
                # If we get any response, server is up
                server_ready = True
                break
            except (httpx.ConnectError, httpx.TimeoutException):
                # Server not ready yet, continue waiting
                pass
            except Exception:
                # Other error - server might be up but returned error
                server_ready = True
                break
            
            time.sleep(check_interval)
        
        if server_ready:
            if self.verbose:
                elapsed = int(time.time() - start_time)
                print(f"[ContainerManager] ✓ llama-server ready on port {port} (took {elapsed}s)")
            return port
        else:
            # Check if container is still running
            if self.is_container_running(model_type):
                if self.verbose:
                    print(f"[ContainerManager] ⚠ Server not ready after {max_wait}s, but container is still running")
                    print(f"[ContainerManager] Server may still be loading model - returning port anyway")
                    print(f"[ContainerManager] HTTP calls will retry automatically")
                return port
            else:
                if self.verbose:
                    print(f"[ContainerManager] ❌ Container crashed while waiting for server")
                    print(f"[ContainerManager] Server will not be available")
                return None
    
    def stop_model_container(self, model_type: str) -> bool:
        """
        Stop Docker container for given model.
        
        Args:
            model_type: Model type ("llama", "phi-3", "gemma")
            
        Returns:
            True if stopped successfully
        """
        container_name = self.MODEL_CONTAINERS.get(model_type)
        if not container_name:
            return False
        
        if not self.is_container_running(model_type):
            if self.verbose:
                print(f"[ContainerManager] Container {container_name} not running")
            return True
        
        if self.verbose:
            print(f"[ContainerManager] Stopping container {container_name}...")
        
        stdout, stderr, returncode = self._run_command(
            ["docker", "stop", container_name],
            check=False
        )
        
        if returncode == 0:
            if model_type in self.running_containers:
                del self.running_containers[model_type]
            if self.verbose:
                print(f"[ContainerManager] Container stopped")
            return True
        else:
            if self.verbose:
                print(f"[ContainerManager] Failed to stop container: {stderr}")
            return False
    
    def stop_all_containers(self):
        """
        Stop all containers that were started by this manager instance.
        Only stops containers tracked in running_containers (i.e., containers we actually started).
        """
        for model_type in list(self.running_containers.keys()):
            self.stop_model_container(model_type)

