#!/usr/bin/env python3
"""
Simple example demonstrating HCOT problem decomposition.
Uses DeepMind Mathematics Dataset for examples.
"""
from hcot_decomposer import quick_decompose
from utils import print_statistics, print_execution_plan
import os

def load_deepmind_examples(num_examples=3):
    """
    Load examples from DeepMind Mathematics Dataset.
    
    The dataset is generated locally, not from HuggingFace.
    Install: git clone https://github.com/deepmind/mathematics_dataset
    Generate: python -m mathematics_dataset.generate --filter=algebra__linear_1d
    """
    # Method 1: Try loading pre-generated dataset from local directory
    local_path = "mathematics_dataset/data/train-easy/algebra__linear_1d.txt"
    if os.path.exists(local_path):
        print(f"Loading {num_examples} examples from local file...")
        examples = []
        with open(local_path, 'r') as f:
            lines = f.readlines()
            for i in range(0, len(lines), 2):
                if len(examples) >= num_examples:
                    break
                if i+1 < len(lines):
                    question = lines[i].strip()
                    answer = lines[i+1].strip()
                    examples.append((question, answer))
        return examples if examples else None
    
    # Method 2: Use hardcoded examples from DeepMind dataset
    print("Using hardcoded DeepMind Mathematics Dataset examples...")
    examples = [
        ("Solve -42*r + 27*c = -1167 and 130*r + 4*c = 372 for r.", "4"),
        ("What is the value of (-3)/(-1)*(-1)/(-3)?", "1"),
        ("Let f(x) = 3*x + 5. What is f(2)?", "11"),
    ]
    return examples[:num_examples]


def check_ollama_setup():
    """Check if Ollama is properly set up and suggest fixes."""
    import socket
    
    # First check if port 11434 is open (Ollama default)
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('127.0.0.1', 11434))
        sock.close()
        
        if result == 0:
            print("✓ Ollama daemon is running on port 11434")
        else:
            print("✗ Cannot connect to Ollama on port 11434")
            print("  Start with: ollama serve")
            return False, None
    except Exception as e:
        print(f"✗ Socket check failed: {e}")
    
    # Now try the ollama library
    try:
        import ollama
        try:
            # Try to list models
            models = ollama.list()
            available = [m["model"] for m in models.get("models", [])]
            
            if available:
                print(f"✓ Available models: {', '.join(available[:3])}")
                return True, available[0]
            else:
                print("\n" + "="*70)
                print("⚠ No models found. Please pull a model:")
                print("="*70)
                print("Try:")
                print("  ollama pull llama3.1")
                print("  ollama pull llama3.2")
                print("  ollama pull qwen2.5:7b")
                print("\nThen check: ollama list")
                print("="*70 + "\n")
                return False, None
                
        except Exception as e:
            print(f"✗ Error connecting to Ollama: {e}")
            print("\nTroubleshooting:")
            print("1. Check if Ollama is running: ps aux | grep ollama")
            print("2. Check port: netstat -tlnp | grep 11434")
            print("3. Try: killall ollama && ollama serve")
            return False, None
            
    except ImportError:
        print("✗ ollama package not installed")
        print("Install with: pip install ollama --break-system-packages")
        return False, None


def example_custom_problem(model="llama3.1"):
    """Run with a custom problem."""
    problem = """
    Solve the system of equations:
    2x + 3y = 13
    x - y = 1
    """
    
    print("="*70)
    print("HCOT MODULE 1: Problem Decomposition Example")
    print("="*70)
    print(f"Using model: {model}")
    print(f"\nProblem:\n{problem}")
    
    # Decompose the problem
    print("\nDecomposing problem...")
    try:
        decomp = quick_decompose(
            problem=problem,
            model=model,
            prompts_path="hcot_prompts.json",
            depth=2,
            branching=3,
            verbose=True
        )
        
        # Show the decomposition
        print("\n" + "="*70)
        print("DECOMPOSITION RESULT")
        print("="*70)
        print(decomp.model_dump_json(indent=2))
        
        # Show statistics
        print_statistics(decomp)
        
        # Show execution plan
        print_execution_plan(decomp)
        
        # Show each sub-problem
        print("="*70)
        print("SUB-PROBLEMS")
        print("="*70)
        for node in decomp.nodes:
            print(f"\n[{node.id}] {node.goal}")
            print(f"    Plan: {node.plan}")
            print(f"    Check: {node.suggested_check.value}")
            if node.depends_on:
                print(f"    Depends on: {', '.join(node.depends_on)}")
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check Ollama is running:")
        print("   ps aux | grep ollama")
        print("2. Check port 11434:")
        print("   netstat -tlnp | grep 11434")
        print("3. Restart Ollama:")
        print("   killall ollama")
        print("   ollama serve &")
        print("4. Pull a model:")
        print("   ollama pull llama3.1")


def example_deepmind_dataset(model="llama3.1"):
    """Run with DeepMind Mathematics Dataset."""
    examples = load_deepmind_examples(num_examples=3)
    
    if not examples:
        print("\nNo dataset found. Falling back to custom problem...")
        example_custom_problem(model)
        return
    
    print("="*70)
    print("HCOT MODULE 1: DeepMind Mathematics Dataset Examples")
    print("="*70)
    print(f"Using model: {model}")
    
    for i, (question, answer) in enumerate(examples, 1):
        print(f"\n{'='*70}")
        print(f"EXAMPLE {i}/{len(examples)}")
        print(f"{'='*70}")
        print(f"Question: {question}")
        print(f"Expected Answer: {answer}")
        
        print("\nDecomposing...")
        try:
            decomp = quick_decompose(
                problem=question,
                model=model,
                prompts_path="hcot_prompts.json",
                depth=2,
                branching=3,
                verbose=False
            )
            
            print(f"\n✓ Decomposed into {len(decomp.nodes)} sub-problems")
            
            # Show sub-problems
            for node in decomp.nodes:
                print(f"  [{node.id}] {node.goal[:60]}")
                if node.depends_on:
                    print(f"       Depends on: {', '.join(node.depends_on)}")
            
        except Exception as e:
            print(f"✗ Failed: {e}")


def main():
    """Main entry point."""
    import sys
    
    print("\n" + "="*70)
    print("HCOT MODULE 1 - Setup Check")
    print("="*70)
    
    # Check Ollama setup
    is_ready, default_model = check_ollama_setup()
    
    if not is_ready:
        print("\n⚠ Please set up Ollama first (see instructions above)")
        print("\nQuick fix:")
        print("  # If 'address already in use' error:")
        print("  ps aux | grep ollama  # Find the PID")
        print("  kill <PID>            # Kill it")
        print("  ollama serve &        # Restart")
        print("  ollama pull llama3.1  # Pull a model")
        return
    
    # Use the detected model or default
    model = default_model if default_model else "llama3.1"
    
    # Check if user wants DeepMind dataset or custom problem
    use_dataset = "--dataset" in sys.argv or "-d" in sys.argv
    
    # Allow model override
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        if idx + 1 < len(sys.argv):
            model = sys.argv[idx + 1]
    
    print("\n" + "="*70)
    if use_dataset:
        example_deepmind_dataset(model)
    else:
        print("Usage:")
        print("  python example.py                    # Run with custom problem")
        print("  python example.py --dataset          # Run with DeepMind examples")
        print("  python example.py --model llama3.2   # Specify model")
        print("="*70)
        example_custom_problem(model)


if __name__ == "__main__":
    main()