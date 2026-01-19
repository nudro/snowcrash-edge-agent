#!/usr/bin/env python3
"""
Run the agentic agent interactively.
Pass prompts and the agent will decide when to use tools.
"""
import sys
import asyncio
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.simple_agent import SimpleAgent


async def main():
    """Run agent interactively."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Snowcrash Agentic Agent")
    parser.add_argument(
        "--model",
        choices=["phi-3", "llama", "gemma"],
        help="Model type to use: phi-3, llama, or gemma"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Direct path to GGUF model file (overrides --model)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature (default: 0.7)"
    )
    
    args = parser.parse_args()
    
    print("Snowcrash Agentic Agent")
    print("=" * 50)
    print()
    
    # Create agent
    try:
        agent = SimpleAgent(
            model_path=args.model_path,
            model_type=args.model,
            temperature=args.temperature,
            verbose=True
        )
        
        if agent.model_type:
            print(f"[OK] Using model: {agent.model_type}")
        
        if not agent.llm:
            print("[WARNING] No LLM available.")
            print("  To use the agent, download models first:")
            print("  python scripts/download_models.py")
            print()
            print("  Or specify a model with --model phi-3|llama|gemma")
            print("  Or provide a path with --model-path /path/to/model.gguf")
            print()
            print("  The agent will work in tool-calling mode only.")
            print()
        
        # Interactive loop
        print("Agent ready! Enter prompts (or 'quit' to exit):")
        print()
        
        while True:
            try:
                prompt = input("You: ").strip()
                
                if prompt.lower() in ["quit", "exit", "q"]:
                    print("Goodbye!")
                    break
                
                if not prompt:
                    continue
                
                print("\n[AGENT] Agent thinking...\n")
                
                # Run agent
                response = await agent.run(prompt)
                
                print(f"Agent: {response}\n")
                print("-" * 50)
                print()
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n[FAIL] Error: {e}\n")
                import traceback
                traceback.print_exc()
                print()
                
    except Exception as e:
        print(f"[FAIL] Failed to create agent: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

