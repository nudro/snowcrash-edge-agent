#!/usr/bin/env python3
"""
Remote Old-Nano Agent using LlamaIndex

A completely separate agent that uses LlamaIndex to interact with the old-nano
Jetson Inference Service. This does NOT modify or interfere with existing agents.

Usage:
    python3 remote_old_nano_agent.py --old-nano-ip 10.163.1.173 --model-path models/llama3.2/Llama-3.2-3B-Instruct-Q4_K_M.gguf
"""
import sys
import os
import argparse
import httpx
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path (for model paths only)
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try importing each module separately to identify which one fails
LLAMA_INDEX_AVAILABLE = False

# Print diagnostic info
print(f"[DEBUG] Python executable: {sys.executable}")
print(f"[DEBUG] Python version: {sys.version.split()[0]}")
print(f"[DEBUG] Python path: {sys.path[:3]}...")  # Show first 3 paths

# Try LlamaIndex LLM base class for custom HTTP adapter (try multiple import paths)
LLAMA_INDEX_LLM_BASE_AVAILABLE = False
LLMMetadata = None
CompletionResponse = None
CompletionResponseGen = None
LLM = None
ChatMessage = None
MessageRole = None

# Try different import paths for LlamaIndex
try:
    from llama_index.core.llms.types import LLMMetadata, CompletionResponse, CompletionResponseGen
    from llama_index.core.llms.llm import LLM
    from llama_index.core.base.llms.types import ChatMessage, MessageRole
    LLAMA_INDEX_LLM_BASE_AVAILABLE = True
    print("[DEBUG] ✓ Successfully imported LlamaIndex LLM base classes")
except ImportError as e1:
    try:
        # Try alternative import path
        from llama_index.llms.types import LLMMetadata, CompletionResponse, CompletionResponseGen
        from llama_index.llms.llm import LLM
        from llama_index.llms.base import ChatMessage, MessageRole
        LLAMA_INDEX_LLM_BASE_AVAILABLE = True
        print("[DEBUG] ✓ Successfully imported LlamaIndex LLM base classes (alternative path)")
    except ImportError as e2:
        print(f"[DEBUG] ✗ Failed to import LlamaIndex LLM base classes: {e1}, {e2}")

# Try LlamaCPP as fallback
try:
    from llama_index.llms.llama_cpp import LlamaCPP
    LLAMA_CPP_AVAILABLE = True
    print("[DEBUG] ✓ Successfully imported LlamaCPP (fallback to direct Python bindings)")
except ImportError as e:
    LLAMA_CPP_AVAILABLE = False
    print(f"[DEBUG] ✗ Failed to import LlamaCPP: {e}")

if not LLAMA_INDEX_LLM_BASE_AVAILABLE and not LLAMA_CPP_AVAILABLE:
    print(f"[ERROR] ✗ No LlamaIndex LLM available")
    print(f"[INFO] Python: {sys.executable}")
    print(f"[INFO] Try: {sys.executable} -m pip install llama-index")
    sys.exit(1)

try:
    from llama_index.core.agent import ReActAgent
    print("[DEBUG] ✓ Successfully imported ReActAgent")
except ImportError as e:
    print(f"[ERROR] ✗ Failed to import ReActAgent: {e}")
    print(f"[INFO] Python: {sys.executable}")
    print(f"[INFO] Try: {sys.executable} -m pip install llama-index")
    print(f"[INFO] Installed packages check:")
    try:
        import subprocess
        result = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True, timeout=5)
        llama_packages = [line for line in result.stdout.split('\n') if 'llama' in line.lower()]
        if llama_packages:
            print("[INFO] Found llama packages:")
            for pkg in llama_packages[:5]:
                print(f"  {pkg}")
        else:
            print("[INFO] No llama packages found in this Python environment")
    except Exception:
        pass
    sys.exit(1)

try:
    from llama_index.core.tools import FunctionTool
    print("[DEBUG] ✓ Successfully imported FunctionTool")
except ImportError as e:
    print(f"[ERROR] ✗ Failed to import FunctionTool: {e}")
    print(f"[INFO] Python: {sys.executable}")
    print(f"[INFO] Try: {sys.executable} -m pip install llama-index")
    sys.exit(1)

try:
    from llama_index.core.query_engine import AgentRunner
    print("[DEBUG] ✓ Successfully imported AgentRunner (optional)")
except ImportError:
    pass  # AgentRunner is optional

LLAMA_INDEX_AVAILABLE = True
print("[DEBUG] All llama-index imports successful!")


def kill_gpu_processes_and_clear_cache(password: str = 'baldur123'):
    """
    Kill any existing GPU processes and clear GPU cache to free memory.
    
    Args:
        password: Password for sudo commands (default: 'baldur123')
    """
    print("[AGENT] Clearing GPU memory and cache...")
    current_pid = os.getpid()
    
    try:
        # Find processes using GPU devices
        gpu_pids = set()
        
        # Method 1: Use fuser to find processes using GPU devices
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
            print(f"[AGENT] fuser method failed: {e}")
        
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
                    if 'python' in line_lower and any(keyword in line_lower for keyword in ['yolo', 'main', 'tracking', 'tensorrt', 'trt', 'cuda', 'llama']):
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
        
        # Kill found processes
        if gpu_pids:
            print(f"[AGENT] Found {len(gpu_pids)} GPU processes to kill: {gpu_pids}")
            killed_count = 0
            for pid in gpu_pids:
                try:
                    # First try with sudo (more reliable)
                    kill_cmd = f'echo "{password}" | sudo -S kill -9 {pid} 2>/dev/null'
                    result = subprocess.run(kill_cmd, shell=True, check=False,
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=2)
                    if result.returncode == 0:
                        killed_count += 1
                        print(f"[AGENT] Killed PID {pid} with sudo")
                    else:
                        # Fallback to normal kill
                        try:
                            os.kill(pid, 9)  # SIGKILL
                            killed_count += 1
                            print(f"[AGENT] Killed PID {pid}")
                        except (ProcessLookupError, PermissionError):
                            pass  # Process already dead or no permission
                except Exception as e:
                    print(f"[AGENT] Failed to kill PID {pid}: {e}")
        
        if killed_count > 0:
            print(f"[AGENT] Killed {killed_count} GPU processes")
            # Wait a moment for processes to die
            import time
            time.sleep(1)
        else:
            print("[AGENT] No GPU processes found to kill")
        
        # Clear GPU cache using nvidia-smi
        try:
            # Try to reset GPU (may not work on all systems)
            try:
                clear_cmd = f'echo "{password}" | sudo -S nvidia-smi --gpu-reset 2>/dev/null || true'
                subprocess.run(clear_cmd, shell=True, timeout=5, capture_output=True)
            except Exception:
                pass
            
            # Clear system page cache (helps free GPU memory)
            try:
                clear_cuda_cmd = f'echo "{password}" | sudo -S sync && echo 3 | sudo -S tee /proc/sys/vm/drop_caches > /dev/null 2>&1 || true'
                subprocess.run(clear_cuda_cmd, shell=True, timeout=5, capture_output=True)
            except Exception:
                pass
            
            print("[AGENT] GPU cache cleared")
        except Exception as e:
            print(f"[AGENT] Warning: Could not clear GPU cache: {e}")
        
        # Additional: Clear Python cache if needed
        try:
            import gc
            gc.collect()
            print("[AGENT] Python garbage collection completed")
        except Exception:
            pass
        
        # Wait a bit more for memory to be fully freed
        import time
        time.sleep(2)
        print("[AGENT] GPU memory clearing complete")
        
    except Exception as e:
        print(f"[AGENT] Error clearing GPU memory: {e}")


class OldNanoClient:
    """Client for interacting with old-nano Jetson Inference Service."""
    
    def __init__(self, host: str = "old-nano", port: int = 9000, password: str = "baldur123", use_ssh: bool = True):
        """
        Initialize old-nano client.
        
        Args:
            host: old-nano hostname or IP address
            port: Service port (default: 9000)
            password: Password for sudo commands on old-nano (default: baldur123)
            use_ssh: Use SSH for system queries instead of HTTP (faster, default: True)
        """
        self.host = host
        self.port = port
        self.password = password
        self.use_ssh = use_ssh
        self.base_url = f"http://{host}:{port}"
        self.client = httpx.Client(timeout=5.0)  # Reduced timeout for faster failures
    
    def _ssh_execute(self, command: str, timeout: float = 3.0) -> str:
        """Execute command on old-nano via SSH."""
        try:
            # Try SSH key-based auth first (no password needed)
            ssh_cmd = f'ssh -o StrictHostKeyChecking=no -o ConnectTimeout=3 -o BatchMode=yes {self.host} "{command}"'
            result = subprocess.run(
                ssh_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            if result.returncode == 0:
                return result.stdout.strip()
            
            # Key auth failed - try password auth with sshpass
            stderr = result.stderr.strip()
            try:
                # Check if sshpass is available
                subprocess.run(['which', 'sshpass'], check=True, capture_output=True, timeout=1, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # Use sshpass for password authentication
                ssh_cmd = f'sshpass -p "{self.password}" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=3 {self.host} "{command}"'
                result = subprocess.run(
                    ssh_cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                if result.returncode == 0:
                    return result.stdout.strip()
                else:
                    error_msg = result.stderr.strip()
                    if "password" in error_msg.lower() or "permission denied" in error_msg.lower():
                        return f"Error: SSH password authentication failed. Check password or set up SSH keys."
                    return f"Error: {error_msg}"
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                # sshpass not available
                return f"Error: SSH key auth failed and sshpass not installed. Install: sudo apt-get install sshpass"
        except subprocess.TimeoutExpired:
            return "Error: SSH timeout (may be waiting for password)"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _sudo_execute(self, command: str, timeout: float = 3.0) -> str:
        """Execute sudo command on old-nano via SSH."""
        # Try password-less sudo first (if configured)
        result = self._ssh_execute(f'sudo {command}', timeout)
        if "Error" not in result or "password" not in result.lower():
            return result
        
        # Fallback to password-based sudo
        sudo_cmd = f'echo "{self.password}" | sudo -S {command}'
        return self._ssh_execute(sudo_cmd, timeout)
    
    def check_health(self) -> Dict[str, Any]:
        """Check if old-nano service is healthy and responding."""
        try:
            response = self.client.get(f"{self.base_url}/health", timeout=3.0)
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException:
            return {"error": "HTTP timeout - service may be slow or unresponsive", "status": "unavailable"}
        except httpx.ConnectError:
            return {"error": "Connection refused - service may not be running", "status": "unavailable"}
        except Exception as e:
            return {"error": str(e), "status": "unavailable"}
    
    def get_watch_status(self) -> Dict[str, Any]:
        """Get status of person detection watch on old-nano."""
        try:
            response = self.client.get(f"{self.base_url}/watch_status")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "status": "unavailable"}
    
    def start_person_watch(self) -> Dict[str, Any]:
        """Start continuous person detection on old-nano."""
        try:
            response = self.client.post(f"{self.base_url}/watch_person")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def stop_person_watch(self) -> Dict[str, Any]:
        """Stop continuous person detection on old-nano."""
        try:
            response = self.client.post(f"{self.base_url}/stop_watch")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get comprehensive information about old-nano service."""
        health = self.check_health()
        watch_status = self.get_watch_status()
        
        return {
            "service_health": health,
            "person_detection": watch_status,
            "base_url": self.base_url,
            "endpoints": {
                "health": f"{self.base_url}/health",
                "video_feed": f"{self.base_url}/video_feed",
                "watch_person": f"{self.base_url}/watch_person",
                "stop_watch": f"{self.base_url}/stop_watch",
                "watch_status": f"{self.base_url}/watch_status"
            }
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information (CPU, memory, disk, etc.) from old-nano."""
        if self.use_ssh:
            # Use SSH for faster direct queries
            try:
                cpu_usage = self._ssh_execute("top -bn1 | grep 'Cpu(s)' | awk '{print $2}' | cut -d'%' -f1", timeout=2.0)
                # Check if SSH returned an error (password issue, etc.)
                if cpu_usage.startswith("Error:"):
                    return {"error": cpu_usage, "source": "ssh", "note": "SSH connection issue - check password/keys"}
                
                mem_info = self._ssh_execute("free -h | grep Mem | awk '{print $3\"/\"$2}'", timeout=2.0)
                disk_info = self._ssh_execute("df -h / | tail -1 | awk '{print $3\"/\"$2\" (\"$5\" used)\"}'", timeout=2.0)
                uptime = self._ssh_execute("uptime -p", timeout=2.0)
                
                return {
                    "cpu_usage": cpu_usage,
                    "memory": mem_info,
                    "disk": disk_info,
                    "uptime": uptime,
                    "source": "ssh"
                }
            except Exception as e:
                return {"error": str(e), "source": "ssh", "note": "SSH exception"}
        
        # Fallback to HTTP
        try:
            response = self.client.get(f"{self.base_url}/system_info", timeout=3.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            try:
                response = self.client.get(f"{self.base_url}/system", timeout=3.0)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                return {"error": str(e), "status": "unavailable"}
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information (memory, temperature, utilization) from old-nano."""
        if self.use_ssh:
            # Use SSH for faster direct queries
            try:
                gpu_mem = self._sudo_execute("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits", timeout=2.0)
                gpu_temp = self._sudo_execute("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits", timeout=2.0)
                gpu_util = self._sudo_execute("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits", timeout=2.0)
                
                return {
                    "memory": gpu_mem,
                    "temperature": gpu_temp,
                    "utilization": gpu_util,
                    "source": "ssh"
                }
            except Exception as e:
                return {"error": str(e), "source": "ssh"}
        
        # Fallback to HTTP
        try:
            response = self.client.get(f"{self.base_url}/gpu_info", timeout=3.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            try:
                response = self.client.get(f"{self.base_url}/gpu", timeout=3.0)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                return {"error": str(e), "status": "unavailable"}
    
    def get_temperature(self) -> Dict[str, Any]:
        """Get system and GPU temperature from old-nano."""
        if self.use_ssh:
            # Use SSH for faster direct queries
            try:
                cpu_temp = self._ssh_execute("cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | head -1 | awk '{print $1/1000\"°C\"}'", timeout=2.0)
                gpu_temp = self._sudo_execute("nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits", timeout=2.0)
                
                return {
                    "cpu": cpu_temp,
                    "gpu": f"{gpu_temp}°C" if gpu_temp and "Error" not in gpu_temp else "N/A",
                    "source": "ssh"
                }
            except Exception as e:
                return {"error": str(e), "source": "ssh"}
        
        # Fallback to HTTP
        try:
            response = self.client.get(f"{self.base_url}/temperature", timeout=3.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            try:
                response = self.client.get(f"{self.base_url}/temp", timeout=3.0)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                return {"error": str(e), "status": "unavailable"}
    
    def get_cpu_info(self) -> Dict[str, Any]:
        """Get CPU information (usage, cores, frequency) from old-nano."""
        if self.use_ssh:
            # Use SSH for faster direct queries
            try:
                cpu_usage = self._ssh_execute("top -bn1 | grep 'Cpu(s)' | awk '{print $2}' | cut -d'%' -f1", timeout=2.0)
                cpu_cores = self._ssh_execute("nproc", timeout=2.0)
                cpu_freq = self._ssh_execute("lscpu | grep 'CPU max MHz' | awk '{print $4}'", timeout=2.0)
                load_avg = self._ssh_execute("uptime | awk -F'load average:' '{print $2}'", timeout=2.0)
                
                return {
                    "usage": f"{cpu_usage}%",
                    "cores": cpu_cores,
                    "frequency": f"{cpu_freq} MHz" if cpu_freq and "Error" not in cpu_freq else "N/A",
                    "load_average": load_avg.strip() if load_avg else "N/A",
                    "source": "ssh"
                }
            except Exception as e:
                return {"error": str(e), "source": "ssh"}
        
        # Fallback to HTTP
        try:
            response = self.client.get(f"{self.base_url}/cpu_info", timeout=3.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            try:
                response = self.client.get(f"{self.base_url}/cpu", timeout=3.0)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                return {"error": str(e), "status": "unavailable"}
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory information (RAM usage, swap) from old-nano."""
        if self.use_ssh:
            # Use SSH for faster direct queries
            try:
                mem_total = self._ssh_execute("free -h | grep Mem | awk '{print $2}'", timeout=2.0)
                mem_used = self._ssh_execute("free -h | grep Mem | awk '{print $3}'", timeout=2.0)
                mem_free = self._ssh_execute("free -h | grep Mem | awk '{print $4}'", timeout=2.0)
                swap_info = self._ssh_execute("free -h | grep Swap | awk '{print $3\"/\"$2}'", timeout=2.0)
                
                return {
                    "total": mem_total,
                    "used": mem_used,
                    "free": mem_free,
                    "swap": swap_info,
                    "source": "ssh"
                }
            except Exception as e:
                return {"error": str(e), "source": "ssh"}
        
        # Fallback to HTTP
        try:
            response = self.client.get(f"{self.base_url}/memory_info", timeout=3.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            try:
                response = self.client.get(f"{self.base_url}/memory", timeout=3.0)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                return {"error": str(e), "status": "unavailable"}


def create_old_nano_tools(old_nano_client: OldNanoClient) -> list:
    """Create LlamaIndex FunctionTools for old-nano system properties only."""
    
    def get_old_nano_system_info() -> str:
        """Get old-nano system info (CPU, memory, disk)."""
        result = old_nano_client.get_system_info()
        if "error" in result:
            error_msg = result.get('error', 'unknown')
            note = result.get('note', '')
            if "password" in error_msg.lower() or "sshpass" in error_msg.lower():
                return f"Error: {error_msg}. {note} Install sshpass: sudo apt-get install sshpass"
            return f"Error: {error_msg}. {note}"
        return str(result)
    
    def get_old_nano_gpu_info() -> str:
        """Get old-nano GPU info (memory, temp, utilization)."""
        result = old_nano_client.get_gpu_info()
        if "error" in result:
            return f"Error: {result.get('error', 'unknown')}"
        return str(result)
    
    def get_old_nano_temperature() -> str:
        """Get old-nano temperatures (CPU, GPU, system)."""
        result = old_nano_client.get_temperature()
        if "error" in result:
            return f"Error: {result.get('error', 'unknown')}"
        return str(result)
    
    def get_old_nano_cpu_info() -> str:
        """Get old-nano CPU info (usage, cores, frequency)."""
        result = old_nano_client.get_cpu_info()
        if "error" in result:
            return f"Error: {result.get('error', 'unknown')}"
        return str(result)
    
    def get_old_nano_memory_info() -> str:
        """Get old-nano memory info (RAM, swap)."""
        result = old_nano_client.get_memory_info()
        if "error" in result:
            return f"Error: {result.get('error', 'unknown')}"
        return str(result)
    
    # Create FunctionTools - System Properties Only
    tools = [
        FunctionTool.from_defaults(fn=get_old_nano_system_info),
        FunctionTool.from_defaults(fn=get_old_nano_gpu_info),
        FunctionTool.from_defaults(fn=get_old_nano_temperature),
        FunctionTool.from_defaults(fn=get_old_nano_cpu_info),
        FunctionTool.from_defaults(fn=get_old_nano_memory_info),
    ]
    
    return tools


def create_agent(
    model_path: str,
    old_nano_host: str = "old-nano",
    old_nano_port: int = 9000,
    temperature: float = 0.1,
    context_window: int = 2048,
    n_gpu_layers: int = 10,
    old_nano_password: str = "baldur123",
    use_ssh: bool = True,
    llama_server_port: int = 8080,
    use_docker_llm: bool = True
) -> ReActAgent:
    """
    Create LlamaIndex ReActAgent with old-nano tools.
    
    Args:
        model_path: Path to GGUF model file
        old_nano_host: old-nano hostname or IP
        old_nano_port: old-nano service port
        temperature: LLM temperature
        context_window: Context window size (reduced for memory)
        n_gpu_layers: Number of GPU layers (default: 10 for memory efficiency)
    
    Returns:
        ReActAgent instance
    """
    if not LLAMA_INDEX_AVAILABLE:
        raise RuntimeError("llama-index not available")
    
    # Initialize old-nano client with SSH support for faster queries
    old_nano_client = OldNanoClient(host=old_nano_host, port=old_nano_port, password=old_nano_password, use_ssh=use_ssh)
    
    # Create tools
    tools = create_old_nano_tools(old_nano_client)
    
    # Initialize LLM - prefer jetson-container llama-server HTTP API (faster, GPU-optimized)
    print(f"[AGENT] use_docker_llm={use_docker_llm}, LLAMA_INDEX_LLM_BASE_AVAILABLE={LLAMA_INDEX_LLM_BASE_AVAILABLE}")
    
    # Initialize llm variable
    llm = None
    
    if use_docker_llm:
        # Create custom LlamaIndex LLM adapter for local llama-server (no internet required)
        print(f"[AGENT] Using jetson-container llama-server HTTP API on port {llama_server_port}")
        print(f"[AGENT] Settings: context={context_window}, max_tokens=128")
        
        try:
            # Always use HTTP API wrapper (works with or without base classes)
            if LLAMA_INDEX_LLM_BASE_AVAILABLE:
                # Use proper LlamaIndex LLM base class if available
                class LocalLlamaServerLLM(LLM):
                    """Custom LlamaIndex LLM that connects to local llama-server HTTP API (no internet required)."""
                    
                    def __init__(self, port: int, temperature: float, max_tokens: int, context_window: int):
                        super().__init__()
                        self.port = port
                        self.temperature = temperature
                        self.max_tokens = max_tokens
                        self.context_window = context_window
                        # Verify localhost to prevent internet calls
                        self._api_url = f"http://localhost:{port}/v1/chat/completions"
                        if not self._api_url.startswith(("http://localhost", "http://127.0.0.1")):
                            raise ValueError(f"API URL must be localhost, got: {self._api_url}")
                        # Use httpx client with explicit localhost - no internet calls
                        self._client = httpx.Client(timeout=30.0)
                    
                    @property
                    def metadata(self) -> LLMMetadata:
                        return LLMMetadata(
                            context_window=self.context_window,
                            num_output=self.max_tokens,
                            model_name="local-llama-server"
                        )
                    
                    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
                        """Synchronous completion."""
                        messages = [{"role": "user", "content": prompt}]
                        payload = {
                            "model": "llama",  # llama-server ignores this but requires it
                            "messages": messages,
                            "temperature": self.temperature,
                            "max_tokens": self.max_tokens,
                        }
                        
                        try:
                            response = self._client.post(
                                self._api_url,
                                json=payload,
                                headers={"Content-Type": "application/json"},
                                timeout=30.0
                            )
                            response.raise_for_status()
                            result = response.json()
                            if "choices" in result and len(result["choices"]) > 0:
                                text = result["choices"][0]["message"]["content"]
                                return CompletionResponse(text=text)
                            else:
                                raise ValueError(f"Unexpected response format: {result}")
                        except Exception as e:
                            raise RuntimeError(f"Error calling local llama-server: {e}")
                    
                    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
                        """Streaming completion (not implemented, returns regular completion)."""
                        response = self.complete(prompt, **kwargs)
                        yield response
                    
                    def chat(self, messages, **kwargs):
                        """Chat interface."""
                        # Convert LlamaIndex messages to OpenAI format
                        openai_messages = []
                        for msg in messages:
                            if isinstance(msg, ChatMessage):
                                role = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
                                openai_messages.append({"role": role, "content": msg.content})
                            else:
                                openai_messages.append({"role": "user", "content": str(msg)})
                        
                        payload = {
                            "model": "llama",
                            "messages": openai_messages,
                            "temperature": self.temperature,
                            "max_tokens": self.max_tokens,
                        }
                        
                        try:
                            response = self._client.post(
                                self._api_url,
                                json=payload,
                                headers={"Content-Type": "application/json"},
                                timeout=30.0
                            )
                            response.raise_for_status()
                            result = response.json()
                            if "choices" in result and len(result["choices"]) > 0:
                                text = result["choices"][0]["message"]["content"]
                                return ChatMessage(role=MessageRole.ASSISTANT, content=text)
                            else:
                                raise ValueError(f"Unexpected response format: {result}")
                        except Exception as e:
                            raise RuntimeError(f"Error calling local llama-server: {e}")
                
                llm = LocalLlamaServerLLM(
                    port=llama_server_port,
                    temperature=temperature,
                    max_tokens=128,
                    context_window=context_window
                )
            else:
                # CRITICAL: OpenAI client validates model names against OpenAI servers (requires internet)
                # We cannot use OpenAI client as it makes internet calls for model validation
                # Must fall back to direct LlamaCPP bindings when base classes aren't available
                print("[AGENT] ⚠️  LlamaIndex base classes not available")
                print("[AGENT] ⚠️  Cannot use HTTP API wrapper (requires base classes)")
                print("[AGENT] ⚠️  OpenAI client would validate models against internet - NOT using it")
                print("[AGENT] Falling back to direct LlamaCPP bindings (100% local, no internet)")
                # Create LlamaCPP instance directly (no HTTP API, but still local-only)
                llm = None  # Will be created in the fallback block below
                use_docker_llm = False
        except Exception as e:
            print(f"[AGENT] Error creating HTTP API wrapper: {e}")
            print(f"[AGENT] Falling back to direct Python bindings...")
            import traceback
            traceback.print_exc()
            # Fall through to direct bindings
            use_docker_llm = False
            llm = None  # Ensure llm is None so we create it below
    
    if not use_docker_llm or llm is None:
        # Fallback to direct Python bindings (slower)
        print(f"[AGENT] Using direct LlamaCPP Python bindings (fallback)")
        print(f"[AGENT] Loading LLM from: {model_path}")
        print(f"[AGENT] Settings: {n_gpu_layers} GPU layers, context={context_window}, max_tokens=128")
        llm = LlamaCPP(
            model_path=model_path,
            temperature=temperature,
            max_new_tokens=128,  # Very short responses for speed
            context_window=context_window,
            model_kwargs={
                "n_gpu_layers": n_gpu_layers,
                "n_ctx": context_window,  # Explicitly set context size
                "n_batch": 512,  # Larger batch for faster processing
                "use_mmap": True,  # Use memory mapping
                "use_mlock": False,  # Don't lock memory
                "n_threads": 4,  # Use more threads for faster inference
            },
            generate_kwargs={
                "stop": ["<|eot_id|>", "<|end_of_text|>"],
                "max_tokens": 128,  # Hard limit on tokens
            },
            verbose=False  # Reduce verbosity
        )
    
    # Create agent
    print(f"[AGENT] Creating ReActAgent with {len(tools)} tools")
    # ReActAgent constructor takes tools and llm as parameters
    try:
        agent = ReActAgent(
            tools=tools,
            llm=llm,
            verbose=True
        )
    except TypeError as e:
        # Try alternative constructor signature
        try:
            agent = ReActAgent.from_tools(
                tools=tools,
                llm=llm,
                verbose=True
            )
        except AttributeError:
            # Last resort: try with different parameter names
            agent = ReActAgent(
                tool=tools,
                llm=llm,
                verbose=True
            )
    
    return agent


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Remote Old-Nano Agent using LlamaIndex"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to GGUF model file (e.g., models/llama3.2/Llama-3.2-3B-Instruct-Q4_K_M.gguf)"
    )
    parser.add_argument(
        "--old-nano-ip",
        type=str,
        default="old-nano",
        help="old-nano hostname or IP address (default: old-nano)"
    )
    parser.add_argument(
        "--old-nano-port",
        type=int,
        default=9000,
        help="old-nano service port (default: 9000)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="LLM temperature (default: 0.1)"
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=2048,
        help="Context window size (default: 2048 for fast responses, use 1536 for less memory)"
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=10,
        help="Number of GPU layers (default: 10 for memory efficiency, use -1 for all layers)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode (chat loop)"
    )
    parser.add_argument(
        "--clear-gpu",
        action="store_true",
        help="Clear GPU memory and kill GPU processes before starting"
    )
    parser.add_argument(
        "--password",
        type=str,
        default="baldur123",
        help="Password for sudo commands on old-nano (default: baldur123)"
    )
    parser.add_argument(
        "--no-ssh",
        action="store_false",
        dest="use_ssh",
        default=True,
        help="Disable SSH, use HTTP only (SSH is enabled by default)"
    )
    parser.add_argument(
        "--llama-server-port",
        type=int,
        default=8080,
        help="Port where jetson-container llama-server is running (default: 8080)"
    )
    parser.add_argument(
        "--no-docker-llm",
        action="store_false",
        dest="use_docker_llm",
        default=True,
        help="Disable Docker LLM (use direct Python bindings instead of jetson-container llama-server)"
    )
    
    args = parser.parse_args()
    
    # Clear GPU memory if requested
    if args.clear_gpu:
        kill_gpu_processes_and_clear_cache(password=args.password)
    
    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        print(f"[INFO] Available models in models/:")
        models_dir = PROJECT_ROOT / "models"
        if models_dir.exists():
            for model_file in models_dir.rglob("*.gguf"):
                print(f"  - {model_file.relative_to(PROJECT_ROOT)}")
        sys.exit(1)
    
    try:
        # Create agent
        agent = create_agent(
            model_path=str(model_path),
            old_nano_host=args.old_nano_ip,
            old_nano_port=args.old_nano_port,
            temperature=args.temperature,
            context_window=args.context_window,
            n_gpu_layers=args.n_gpu_layers,
            old_nano_password=args.password,
            use_ssh=args.use_ssh,
            llama_server_port=args.llama_server_port,
            use_docker_llm=args.use_docker_llm
        )
        
        print("")
        print("=" * 60)
        print("Remote Old-Nano Agent (LlamaIndex)")
        print("=" * 60)
        print(f"Model: {model_path}")
        print(f"old-nano: {args.old_nano_ip}:{args.old_nano_port}")
        print("")
        print("Available tools (System Properties Only):")
        print("  - get_old_nano_system_info: Get CPU, memory, disk usage")
        print("  - get_old_nano_gpu_info: Get GPU memory, temperature, utilization")
        print("  - get_old_nano_temperature: Get CPU/GPU/system temperatures")
        print("  - get_old_nano_cpu_info: Get CPU usage, cores, frequency")
        print("  - get_old_nano_memory_info: Get RAM usage, swap, memory stats")
        print("")
        
        if args.interactive:
            # Interactive mode
            print("Interactive mode - type 'quit' or 'exit' to stop")
            print("")
            while True:
                try:
                    user_input = input("You: ").strip()
                    if not user_input:
                        continue
                    if user_input.lower() in ["quit", "exit", "q"]:
                        break
                    
                    # Use the run() method - handle async event loop properly
                    try:
                        import asyncio
                        import threading
                        
                        # The agent.run() method is synchronous but uses async workflows internally
                        # It needs to be called from within a running event loop
                        response = None
                        exception = None
                        
                        def run_in_thread():
                            nonlocal response, exception
                            try:
                                # Create a new event loop for this thread
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                
                                # Try to use nest_asyncio to allow nested event loops
                                # This allows sync code to use async features when loop is running
                                try:
                                    import nest_asyncio
                                    nest_asyncio.apply(loop)
                                    # Try different methods to call the agent
                                    async def run_agent():
                                        # First try: Use query() method if available (often more reliable)
                                        if hasattr(agent, 'query'):
                                            try:
                                                query_engine = agent.as_query_engine() if hasattr(agent, 'as_query_engine') else agent
                                                if hasattr(query_engine, 'query'):
                                                    result = await asyncio.to_thread(query_engine.query, user_input)
                                                    return result
                                            except:
                                                pass
                                        
                                        # Second try: Use run() method
                                        workflow_handler = agent.run(user_input)
                                        # If result is a workflow handler, we need to wait for it
                                        if hasattr(workflow_handler, 'result'):
                                            # Wait for the workflow to complete
                                            # The result() method might be async or need waiting
                                            try:
                                                # Try to get result - check if it's a coroutine
                                                if asyncio.iscoroutinefunction(workflow_handler.result):
                                                    return await workflow_handler.result()
                                                else:
                                                    # Sync result - but might not be ready
                                                    # Wait longer and try again
                                                    import time
                                                    max_wait = 30  # Wait up to 30 seconds
                                                    waited = 0
                                                    for attempt in range(150):  # Try up to 150 times (30 seconds)
                                                        try:
                                                            result = workflow_handler.result()
                                                            if result is not None:
                                                                return result
                                                        except Exception as e:
                                                            error_str = str(e).lower()
                                                            # If "not set" or "not ready", keep waiting
                                                            if "not set" in error_str or "not ready" in error_str or "not available" in error_str:
                                                                if attempt < 149:  # Not last attempt
                                                                    await asyncio.sleep(0.2)
                                                                    waited += 0.2
                                                                    continue
                                                            # Other error or last attempt - try alternative methods
                                                            break
                                                    # If still not ready, try alternative attributes
                                                    if hasattr(workflow_handler, 'response'):
                                                        return workflow_handler.response
                                                    if hasattr(workflow_handler, 'output'):
                                                        return workflow_handler.output
                                                    if hasattr(workflow_handler, 'get_response'):
                                                        try:
                                                            return workflow_handler.get_response()
                                                        except:
                                                            pass
                                                    return workflow_handler
                                            except Exception as e:
                                                # Can't get result, return handler
                                                return workflow_handler
                                        return workflow_handler
                                    response = loop.run_until_complete(run_agent())
                                except ImportError:
                                    # nest_asyncio not available - try direct call
                                    # The agent.run() needs a running loop, so start loop first
                                    async def run_agent():
                                        # Call the sync method - nest_asyncio would help here
                                        # Without it, we need to run in executor but set loop there
                                        def run_with_loop():
                                            # Set the loop in this thread
                                            asyncio.set_event_loop(loop)
                                            workflow_handler = agent.run(user_input)
                                            # Try to extract result if it's a workflow handler
                                            if hasattr(workflow_handler, 'result'):
                                                try:
                                                    # Wait for workflow to complete
                                                    import time
                                                    max_wait = 30  # Wait up to 3 seconds
                                                    waited = 0
                                                    while waited < max_wait:
                                                        try:
                                                            result = workflow_handler.result()
                                                            if result is not None:
                                                                return result
                                                        except Exception:
                                                            time.sleep(0.2)
                                                            waited += 0.2
                                                    # Timeout - try to get response attribute
                                                    if hasattr(workflow_handler, 'response'):
                                                        return workflow_handler.response
                                                    return workflow_handler
                                                except Exception:
                                                    if hasattr(workflow_handler, 'response'):
                                                        return workflow_handler.response
                                                    return workflow_handler
                                            return workflow_handler
                                        return await loop.run_in_executor(None, run_with_loop)
                                    response = loop.run_until_complete(run_agent())
                                
                                loop.close()
                            except Exception as e:
                                exception = e
                        
                        # Run in a separate thread
                        thread = threading.Thread(target=run_in_thread)
                        thread.start()
                        thread.join()
                        
                        if exception:
                            raise exception
                        
                        # Extract response text
                        # The response might be a workflow handler that needs result extraction
                        response_text = None
                        import time
                        
                        if hasattr(response, 'result'):
                            try:
                                # Wait longer for workflow to complete (up to 30 seconds)
                                max_retries = 150  # Wait up to 30 seconds (150 * 0.2s)
                                result = None
                                last_error = None
                                
                                for attempt in range(max_retries):
                                    try:
                                        result = response.result()
                                        if result is not None:
                                            break
                                    except Exception as e:
                                        last_error = e
                                        error_str = str(e).lower()
                                        # If "not set" or "not ready", keep waiting
                                        if "not set" in error_str or "not ready" in error_str or "not available" in error_str:
                                            if attempt < max_retries - 1:
                                                time.sleep(0.2)
                                                continue
                                        # Other errors - try alternative methods
                                        break
                                
                                # If result is still None, try alternative methods
                                if result is None:
                                    # Try direct attributes
                                    if hasattr(response, 'response'):
                                        result = response.response
                                    elif hasattr(response, 'output'):
                                        result = response.output
                                    elif hasattr(response, 'source_nodes'):
                                        result = response
                                    elif hasattr(response, 'get_response'):
                                        try:
                                            result = response.get_response()
                                        except:
                                            pass
                                    elif hasattr(response, 'get_result'):
                                        try:
                                            result = response.get_result()
                                        except:
                                            pass
                                
                                # Extract text from result
                                if result is not None:
                                    if isinstance(result, str):
                                        response_text = result
                                    elif hasattr(result, 'response'):
                                        response_text = result.response
                                    elif hasattr(result, 'output'):
                                        response_text = result.output
                                    elif hasattr(result, 'text'):
                                        response_text = result.text
                                    elif hasattr(result, 'content'):
                                        response_text = result.content
                                    elif hasattr(result, 'source_nodes'):
                                        # Try to extract from source nodes
                                        try:
                                            response_text = str(result)
                                        except:
                                            response_text = "Response available but cannot extract text"
                                    else:
                                        response_text = str(result)
                                else:
                                    # No result after waiting - try to get error or status
                                    if last_error:
                                        error_msg = str(last_error)
                                        if "not set" in error_msg.lower():
                                            response_text = f"Workflow completed but result not available. Try checking workflow status or retry the query."
                                        else:
                                            response_text = f"Workflow error: {error_msg}"
                                    else:
                                        response_text = "Workflow completed but no result available. The agent may still be processing."
                                        
                            except Exception as e:
                                # Result extraction failed - try alternative methods
                                error_msg = str(e)
                                if hasattr(response, 'response'):
                                    response_text = response.response
                                elif hasattr(response, 'output'):
                                    response_text = response.output
                                elif hasattr(response, 'source_nodes'):
                                    response_text = str(response)
                                elif hasattr(response, 'status'):
                                    response_text = f"Workflow status: {response.status}. Error: {error_msg}"
                                else:
                                    response_text = f"Error extracting response: {error_msg}. Try retrying the query."
                        elif hasattr(response, 'response'):
                            response_text = response.response
                        elif hasattr(response, 'output'):
                            response_text = response.output
                        elif hasattr(response, 'source_nodes'):
                            # Response object with source nodes
                            response_text = str(response)
                        elif isinstance(response, str):
                            response_text = response
                        else:
                            # Try to convert to string, but handle workflow handlers carefully
                            try:
                                # Don't call str() on workflow handlers directly
                                if hasattr(response, '__class__') and 'workflow' in str(response.__class__).lower():
                                    response_text = "Workflow handler - result not available"
                                else:
                                    response_text = str(response)
                            except Exception as e:
                                response_text = f"Error extracting response: {e}"
                        
                        if not response_text:
                            response_text = "No response generated"
                        
                        print(f"Agent: {response_text}")
                            
                    except Exception as e:
                        print(f"[ERROR] Error calling agent: {e}")
                        import traceback
                        traceback.print_exc()
                    print("")
                except KeyboardInterrupt:
                    print("\n[INFO] Interrupted by user")
                    break
                except Exception as e:
                    print(f"[ERROR] {e}")
                    import traceback
                    traceback.print_exc()
        else:
            # Single query mode
            print("Single query mode - provide query as argument or use --interactive")
            print("Example queries:")
            print("  - 'get old nano cpu info'")
            print("  - 'get old memory info'")
            print("  - 'get old nano gpu info'")
            print("  - 'get old nano temperature'")
            print("  - 'get old nano system info'")
            print("  - 'what's the CPU usage on old-nano?'")
            print("  - 'how hot is old-nano?'")
            print("  - 'check old-nano health'")
            print("")
            print("Use --interactive flag for chat mode")
            print("Or provide query as argument: python3 remote_old_nano_agent.py --model-path <path> 'get old nano cpu info'")
    
    except Exception as e:
        print(f"[ERROR] Failed to create agent: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

