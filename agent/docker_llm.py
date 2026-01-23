"""
Docker LLM Adapter - Wraps llama-server HTTP API for use with LangChain.

This adapter allows SimpleAgent to use llama.cpp running in Docker containers
instead of llama-cpp-python, providing better memory management and isolation.
"""
import asyncio
import json
import time
from typing import Optional, List, Dict, Any
import httpx
from langchain_core.language_models.llms import BaseLLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult, Generation
from pydantic import Field, PrivateAttr


class DockerLLMAdapter(BaseLLM):
    """
    LangChain-compatible adapter for llama-server running in Docker.
    
    Connects to llama-server HTTP API (OpenAI-compatible endpoint).
    """
    
    # Pydantic field declarations (required for BaseLLM)
    model_type: str = Field(description="Model type identifier")
    port: int = Field(default=8080, description="Port where llama-server is running")
    temperature: float = Field(default=0.7, description="LLM temperature")
    timeout: int = Field(default=60, description="HTTP request timeout in seconds")
    verbose: bool = Field(default=False, description="Print debug messages")
    
    # Non-Pydantic fields (internal state, not serialized)
    _api_url: Optional[str] = PrivateAttr(default=None)
    # Use sync client to avoid event loop issues across threads
    _client: Optional[httpx.Client] = PrivateAttr(default=None)
    
    def __init__(
        self,
        model_type: str,
        port: int = 8080,
        temperature: float = 0.7,
        timeout: int = 60,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize Docker LLM adapter.
        
        Args:
            model_type: Model type ("llama", "phi-3", "gemma")
            port: Port where llama-server is running
            temperature: LLM temperature
            timeout: HTTP request timeout in seconds
            verbose: Print debug messages
        """
        super().__init__(
            model_type=model_type,
            port=port,
            temperature=temperature,
            timeout=timeout,
            verbose=verbose,
            **kwargs
        )
        
        # HTTP API endpoint (OpenAI-compatible)
        self._api_url = f"http://localhost:{port}/v1/chat/completions"
        
        # Use sync HTTP client to avoid event loop issues when called from Flask threads
        # Sync client works in all contexts (sync, async, threads) without event loop conflicts
        self._client = httpx.Client(timeout=self.timeout)
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """
        Generate responses for prompts (required by BaseLLM).
        
        Args:
            prompts: List of input prompts
            stop: Stop sequences
            run_manager: Callback manager
            system_prompt: Optional system prompt (for prompt templates)
            **kwargs: Additional arguments
            
        Returns:
            LLMResult with generations
        """
        # For simplicity, process first prompt (most common case)
        # Can be extended to handle multiple prompts if needed
        prompt = prompts[0] if prompts else ""
        text = self._call(prompt, stop=stop, run_manager=run_manager, system_prompt=system_prompt, **kwargs)
        
        # Create LLMResult with single generation
        generations = [[Generation(text=text)]]
        return LLMResult(generations=generations)
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Synchronous call - uses sync HTTP client (no event loop issues).
        """
        return self._sync_call(prompt, stop, run_manager, system_prompt=system_prompt, **kwargs)
    
    def _sync_call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Synchronous HTTP call to llama-server (using sync httpx.Client).
        Avoids event loop issues when called from Flask threads.
        """
        if self.verbose:
            print(f"[DockerLLM] Calling llama-server on port {self.port}")
            print(f"[DockerLLM] Prompt: {prompt[:100]}...")
        
        # Build messages list
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Prepare request payload
        payload = {
            "model": self.model_type,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 512,
        }
        if stop:
            payload["stop"] = stop
        
        # Retry logic
        max_retries = 3
        retry_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                # Make HTTP POST request (sync)
                response = self._client.post(
                    self._api_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                # Parse response
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    text = result["choices"][0]["message"]["content"]
                    if self.verbose:
                        print(f"[DockerLLM] Response: {text[:100]}...")
                    return text
                else:
                    raise ValueError(f"Unexpected response format: {result}")
                    
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                if attempt < max_retries - 1:
                    if self.verbose:
                        print(f"[DockerLLM] Connection attempt {attempt + 1} failed: {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    error_msg = f"HTTP error calling llama-server: All connection attempts failed. Server may not be running on port {self.port}"
                    if self.verbose:
                        print(f"[DockerLLM] ERROR: {error_msg}")
                    raise RuntimeError(error_msg) from e
            except httpx.HTTPStatusError as e:
                error_msg = f"HTTP {e.response.status_code} error calling llama-server: {e.response.text}"
                if self.verbose:
                    print(f"[DockerLLM] ERROR: {error_msg}")
                raise RuntimeError(error_msg) from e
            except Exception as e:
                error_msg = f"Unexpected error: {e}"
                if self.verbose:
                    print(f"[DockerLLM] ERROR: {error_msg}")
                raise RuntimeError(error_msg) from e
    
    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Async call - wraps sync implementation in thread pool (for LangChain compatibility).
        Uses sync client to avoid event loop conflicts.
        """
        # Run sync call in thread pool to avoid blocking event loop
        import concurrent.futures
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            return await loop.run_in_executor(
                executor,
                lambda: self._sync_call(prompt, stop, run_manager, system_prompt=system_prompt, **kwargs)
            )
    
    @property
    def _llm_type(self) -> str:
        """Return LLM type identifier."""
        return f"docker_llama_{self.model_type}"
    
    async def close(self):
        """Close HTTP client (call this when done)."""
        try:
            if hasattr(self, '_client') and self._client:
                self._client.close()
        except:
            pass
    
    def __del__(self):
        """Cleanup HTTP client on deletion."""
        try:
            if hasattr(self, '_client') and self._client:
                self._client.close()
        except:
            pass

