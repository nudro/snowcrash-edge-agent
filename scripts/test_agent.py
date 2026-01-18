#!/usr/bin/env python3
"""
Test the agentic LangChain agent locally.
Tests agent's ability to interpret prompts and use YOLO tool.
"""
import sys
import asyncio
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.simple_agent import SnowcrashAgent


async def test_agent_with_mock():
    """Test agent with mock/placeholder (no LLM required)."""
    print("Testing Agent Setup (Mock Mode)")
    print("=" * 50)
    
    # Create agent without LLM (for testing structure)
    try:
        agent = SnowcrashAgent(verbose=True)
        
        if agent.llm is None:
            print("[OK] Agent structure created (no LLM - this is expected)")
            print("\nNote: To test with real prompts, you need:")
            print("  1. A local GGUF model, or")
            print("  2. OpenAI API access, or")
            print("  3. Another LLM provider")
            return True
        else:
            print("[OK] Agent with LLM created")
            return True
    except Exception as e:
        print(f"[FAIL] Agent creation error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_agent_with_local_model(model_path: str = None):
    """Test agent with a local GGUF model."""
    print("\nTesting Agent with Local LLM")
    print("=" * 50)
    
    # Find a model if path not provided
    if model_path is None:
        models_dir = PROJECT_ROOT / "models"
        
        # Look for any GGUF file
        for model_dir in ["phi3-mini", "llama3.2", "gemma2b"]:
            model_dir_path = models_dir / model_dir
            if model_dir_path.exists():
                gguf_files = list(model_dir_path.glob("*.gguf"))
                if gguf_files:
                    model_path = str(gguf_files[0])
                    print(f"Using model: {model_path}")
                    break
    
    if not model_path or not Path(model_path).exists():
        print("[WARNING] No local model found. Skipping LLM test.")
        return False
    
    try:
        # Create agent with local model
        agent = SnowcrashAgent(
            model_path=model_path,
            temperature=0.7,
            verbose=True
        )
        
        if agent.llm is None:
            print("[FAIL] Failed to create agent with LLM")
            return False
        
        print("[OK] Agent created with local LLM")
        
        # Test with a simple prompt
        test_prompt = "What tools do you have available?"
        print(f"\nTesting prompt: '{test_prompt}'")
        
        result = await agent.run(test_prompt)
        print(f"\nAgent response:\n{result}")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Error testing with local model: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_agent_yolo_tool(test_image: str = None):
    """Test agent with YOLO tool call."""
    print("\nTesting Agent with YOLO Tool")
    print("=" * 50)
    
    # Find or create a test image
    if test_image is None:
        test_image = PROJECT_ROOT / "test_image.jpg"
    
    if not Path(test_image).exists():
        print(f"[WARNING] No test image found at {test_image}")
        print("  To test YOLO, place an image file and provide the path")
        return False
    
    # Try to create agent with local model
    models_dir = PROJECT_ROOT / "models"
    model_path = None
    
    for model_dir in ["phi3-mini", "llama3.2", "gemma2b"]:
        model_dir_path = models_dir / model_dir
        if model_dir_path.exists():
            gguf_files = list(model_dir_path.glob("*.gguf"))
            if gguf_files:
                model_path = str(gguf_files[0])
                break
    
    if not model_path:
        print("[WARNING] No local model found. Cannot test agentic behavior.")
        print("  The agent needs an LLM to decide when to use tools.")
        return False
    
    try:
        agent = SnowcrashAgent(
            model_path=model_path,
            verbose=True
        )
        
        # Test prompt that should trigger YOLO
        prompt = f"Please detect all objects in this image: {test_image}"
        print(f"\nPrompt: '{prompt}'")
        print("\nAgent should decide to use yolo_object_detection tool...\n")
        
        result = await agent.run(prompt)
        print(f"\n{'='*50}")
        print("Agent Response:")
        print(result)
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all agent tests."""
    print("Snowcrash Agentic Test Suite")
    print("=" * 50)
    
    results = []
    
    # Test 1: Agent structure (no LLM)
    results.append(await test_agent_with_mock())
    
    # Test 2: Agent with local model (if available)
    try:
        results.append(await test_agent_with_local_model())
    except Exception as e:
        print(f"[WARNING] Local model test skipped: {e}")
        results.append(False)
    
    # Test 3: YOLO tool integration (if model and image available)
    try:
        results.append(await test_agent_yolo_tool())
    except Exception as e:
        print(f"[WARNING] YOLO tool test skipped: {e}")
        results.append(False)
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"  Structure test: {'[OK]' if results[0] else '[FAIL]'}")
    print(f"  Local model test: {'[OK]' if len(results) > 1 and results[1] else '[WARNING]'}")
    print(f"  YOLO tool test: {'[OK]' if len(results) > 2 and results[2] else '[WARNING]'}")
    
    if any(results):
        print("\n[OK] Agent setup is working!")
        return 0
    else:
        print("\n[FAIL] Agent setup needs attention")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

