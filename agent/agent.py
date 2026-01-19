#!/usr/bin/env python3
"""
Agentic LangChain agent with MCP tools integration.
Uses an LLM to interpret prompts and decide when to use tools.
"""
import sys
import asyncio
from pathlib import Path
from typing import List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain.agents import create_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough

# Import LangChain tools
from agent.langchain_tools import get_langchain_tools

# For local LLM (llama.cpp)
try:
    from langchain_community.llms import LlamaCpp
    from langchain_community.chat_models import ChatLlamaCpp
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

# For OpenAI-compatible API (if available)
try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class SnowcrashAgent:
    """Agentic LLM agent with MCP tools."""
    
    def __init__(
        self,
        llm=None,
        model_path: Optional[str] = None,
        temperature: float = 0.7,
        verbose: bool = True
    ):
        """
        Initialize the agent.
        
        Args:
            llm: Pre-initialized LLM (optional)
            model_path: Path to local GGUF model (optional)
            temperature: LLM temperature
            verbose: Whether to print agent reasoning
        """
        self.verbose = verbose
        self.tools = get_langchain_tools()
        
        # Initialize LLM
        if llm is None:
            if model_path and LLAMA_CPP_AVAILABLE:
                # Use local GGUF model
                self.llm = self._create_llama_llm(model_path, temperature)
            elif OPENAI_AVAILABLE:
                # Try OpenAI-compatible API (may require API key)
                try:
                    self.llm = ChatOpenAI(temperature=temperature)
                except:
                    # Fallback to a mock LLM for testing
                    print("[WARNING] Warning: No LLM available. Using mock agent.")
                    self.llm = None
            else:
                print("[WARNING] Warning: No LLM available. Install llama-cpp-python or langchain-openai")
                self.llm = None
        else:
            self.llm = llm
        
        # Create agent if LLM is available
        if self.llm:
            self.agent_executor = self._create_agent()
        else:
            self.agent_executor = None
    
    def _create_llama_llm(self, model_path: str, temperature: float):
        """Create a LlamaCpp LLM from GGUF file."""
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama-cpp-python not available. Install with: pip install llama-cpp-python")
        
        from langchain_community.llms import LlamaCpp
        
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        return LlamaCpp(
            model_path=str(model_path),
            temperature=temperature,
            n_ctx=2048,  # Context window
            n_batch=512,  # Batch size
            verbose=False,
            n_gpu_layers=0,  # CPU only for now (can be configured)
        )
    
    def _create_agent(self):
        """Create LangChain agent with tools."""
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant with access to tools for object detection in images.
When the user asks you to analyze an image, detect objects, or identify what's in a picture, 
use the yolo_object_detection tool.

Be concise and helpful. Always explain what you found in the image.
If you cannot detect objects or there's an error, explain what went wrong."""),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Use create_agent from langchain.agents (newer API)
        try:
            # Try new API
            agent = create_agent(
                self.llm,
                self.tools,
                prompt,
                verbose=self.verbose
            )
            return agent
        except Exception as e:
            # Fallback: Create simple tool-calling agent
            print(f"[WARNING] Warning: Using fallback agent creation: {e}")
            
            # For llama.cpp models, we'll create a simpler executor
            from langchain.agents import AgentExecutor
            from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
            from langchain_core.agents import AgentAction, AgentFinish
            
            # Simple agent that can call tools
            class SimpleToolAgent:
                def __init__(self, llm, tools):
                    self.llm = llm
                    self.tools = {tool.name: tool for tool in tools}
                
                async def ainvoke(self, inputs):
                    prompt_text = inputs["input"]
                    # For now, return a simple response
                    # In production, this would use the LLM to decide tool usage
                    return {"output": f"Received: {prompt_text}. Tools available: {list(self.tools.keys())}"}
            
            agent = SimpleToolAgent(self.llm, self.tools)
            executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=self.verbose,
                handle_parsing_errors=True,
                max_iterations=5
            )
            return executor
    
    async def run(self, prompt: str, chat_history: Optional[List] = None) -> str:
        """Run the agent with a prompt."""
        if self.agent_executor is None:
            return "Error: No LLM available. Cannot run agent."
        
        try:
            # Prepare input
            inputs = {"input": prompt}
            if chat_history:
                inputs["chat_history"] = chat_history
            
            # Run agent
            if asyncio.iscoroutinefunction(self.agent_executor.ainvoke):
                result = await self.agent_executor.ainvoke(inputs)
            else:
                result = await asyncio.to_thread(self.agent_executor.invoke, inputs)
            
            return result.get("output", str(result))
            
        except Exception as e:
            error_msg = f"Agent error: {str(e)}"
            if self.verbose:
                import traceback
                traceback.print_exc()
            return error_msg
    
    def run_sync(self, prompt: str, chat_history: Optional[List] = None) -> str:
        """Synchronous wrapper for run."""
        return asyncio.run(self.run(prompt, chat_history))

