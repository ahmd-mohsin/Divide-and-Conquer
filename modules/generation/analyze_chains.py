#!/usr/bin/env python3
"""
Analyze Chain Generation Results
Provides statistics and insights about generated chains.
"""
import json
from pathlib import Path
from typing import Dict, Any, List
import sys


def load_chain_dataset(output_dir: str = "data/chains") -> tuple:
    """Load chain dataset metadata and index."""
    output_path = Path(output_dir)
    
    with open(output_path / "chain_dataset_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    with open(output_path / "chain_dataset_index.json", 'r') as f:
        index = json.load(f)
    
    return metadata, index


def analyze_chains(output_dir: str = "data/chains", verbose: bool = True) -> Dict[str, Any]:
    """Analyze generated chains and provide statistics."""
    
    metadata, index = load_chain_dataset(output_dir)
    
    if not index:
        print("No problems found in dataset")
        return {}
    
    # Overall statistics
    stats = {
        "total_problems": len(index),
        "total_subproblems": metadata["total_subproblems"],
        "total_chains": metadata["total_chains"],
        "avg_chains_per_problem": metadata["total_chains"] / len(index),
        "avg_subproblems_per_problem": metadata["total_subproblems"] / len(index),
        "datasets": metadata["datasets"],
        "models": metadata["models"]
    }
    
    # Reward statistics
    all_rewards = []
    all_success_rates = []
    by_dataset = {}
    
    for entry in index:
        all_rewards.append(entry["avg_reward"])
        all_success_rates.append(entry["success_rate"])
        
        dataset = entry["dataset"]
        if dataset not in by_dataset:
            by_dataset[dataset] = {
                "count": 0,
                "rewards": [],
                "success_rates": []
            }
        
        by_dataset[dataset]["count"] += 1
        by_dataset[dataset]["rewards"].append(entry["avg_reward"])
        by_dataset[dataset]["success_rates"].append(entry["success_rate"])
    
    stats["avg_reward"] = sum(all_rewards) / len(all_rewards)
    stats["avg_success_rate"] = sum(all_success_rates) / len(all_success_rates)
    stats["min_reward"] = min(all_rewards)
    stats["max_reward"] = max(all_rewards)
    
    # By dataset
    stats["by_dataset"] = {}
    for dataset, data in by_dataset.items():
        stats["by_dataset"][dataset] = {
            "count": data["count"],
            "avg_reward": sum(data["rewards"]) / len(data["rewards"]),
            "avg_success_rate": sum(data["success_rates"]) / len(data["success_rates"])
        }
    
    if verbose:
        print_statistics(stats)
    
    return stats


def print_statistics(stats: Dict[str, Any]):
    """Print formatted statistics."""
    
    print("\n" + "="*70)
    print("CHAIN GENERATION STATISTICS")
    print("="*70)
    
    print(f"\nOverall:")
    print(f"  Total problems: {stats['total_problems']}")
    print(f"  Total subproblems: {stats['total_subproblems']}")
    print(f"  Total chains: {stats['total_chains']}")
    print(f"  Avg subproblems per problem: {stats['avg_subproblems_per_problem']:.1f}")
    print(f"  Avg chains per problem: {stats['avg_chains_per_problem']:.1f}")
    
    print(f"\nReward Statistics:")
    print(f"  Average reward: {stats['avg_reward']:.3f}")
    print(f"  Min reward: {stats['min_reward']:.3f}")
    print(f"  Max reward: {stats['max_reward']:.3f}")
    print(f"  Success rate: {stats['avg_success_rate']:.1%}")
    
    print(f"\nBy Dataset:")
    for dataset, data in stats["by_dataset"].items():
        print(f"  {dataset}:")
        print(f"    Problems: {data['count']}")
        print(f"    Avg reward: {data['avg_reward']:.3f}")
        print(f"    Success rate: {data['avg_success_rate']:.1%}")
    
    print(f"\nDatasets used: {', '.join(stats['datasets'])}")
    print(f"Models used: {', '.join(stats['models'])}")
    
    print("="*70 + "\n")


def find_best_chains(output_dir: str = "data/chains", top_n: int = 5) -> List[Dict[str, Any]]:
    """Find problems with best average rewards."""
    
    _, index = load_chain_dataset(output_dir)
    
    # Sort by average reward
    sorted_problems = sorted(index, key=lambda x: x["avg_reward"], reverse=True)
    
    return sorted_problems[:top_n]


def find_challenging_problems(output_dir: str = "data/chains", top_n: int = 5) -> List[Dict[str, Any]]:
    """Find problems with lowest success rates."""
    
    _, index = load_chain_dataset(output_dir)
    
    # Sort by success rate
    sorted_problems = sorted(index, key=lambda x: x["success_rate"])
    
    return sorted_problems[:top_n]


def export_for_grpo(
    output_dir: str = "data/chains",
    export_file: str = "data/grpo_dataset.jsonl",
    min_reward: float = 0.0
) -> int:
    """
    Export hierarchical chains in format suitable for GRPO training.
    
    Format per line:
    {
        "prompt": "problem context",
        "completions": [
            {"text": "full execution chain 1", "reward": 0.8},
            {"text": "full execution chain 2", "reward": 1.0},
            ...
        ]
    }
    
    Returns:
        Number of problems exported
    """
    
    output_path = Path(output_dir)
    chains_dir = output_path / "chains"
    
    _, index = load_chain_dataset(output_dir)
    
    export_data = []
    
    for entry in index:
        # Load full chain data
        chain_file = output_path / entry["file"]
        with open(chain_file, 'r') as f:
            chain_data = json.load(f)
        
        # Filter chains by minimum reward
        chains = [
            c for c in chain_data.get("chains", [])
            if c.get("final_reward", 0.0) >= min_reward
        ]
        
        if not chains:
            continue
        
        # Format each chain as a single text (all steps concatenated)
        completions = []
        for chain in chains:
            # Build full execution trace
            steps_text = []
            for step in chain.get('steps', []):
                step_text = f"Step: {step['goal']}\nReasoning: {step['reasoning']}\nAnswer: {step['answer']}\n"
                steps_text.append(step_text)
            
            full_chain_text = "\n".join(steps_text)
            if chain.get('final_answer'):
                full_chain_text += f"\nFinal Answer: {chain['final_answer']}"
            
            completions.append({
                "text": full_chain_text,
                "reward": chain.get("final_reward", 0.0)
            })
        
        # Create GRPO format entry
        grpo_entry = {
            "prompt": f"Problem: {chain_data['problem']}\n\nSolve this problem step by step through the following subproblems:",
            "completions": completions,
            "ground_truth": chain_data["ground_truth"],
            "problem_id": entry["id"],
            "num_subproblems": chain_data.get("num_subproblems", 0)
        }
        
        export_data.append(grpo_entry)
    
    # Write to JSONL
    export_path = Path(export_file)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(export_path, 'w') as f:
        for entry in export_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"\n✓ Exported {len(export_data)} problems to {export_file}")
    print(f"  Format: JSONL with hierarchical execution chains")
    print(f"  Ready for GRPO training\n")
    
    return len(export_data)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze chain generation results")
    parser.add_argument("--input-dir", default="data/chains",
                       help="Directory containing chain dataset")
    parser.add_argument("--export-grpo", action="store_true",
                       help="Export data in GRPO format")
    parser.add_argument("--export-file", default="data/grpo_dataset.jsonl",
                       help="Output file for GRPO export")
    parser.add_argument("--min-reward", type=float, default=0.0,
                       help="Minimum reward for exported chains")
    parser.add_argument("--show-best", type=int, default=0,
                       help="Show N best problems")
    parser.add_argument("--show-challenging", type=int, default=0,
                       help="Show N most challenging problems")
    
    args = parser.parse_args()
    
    # Analyze
    stats = analyze_chains(args.input_dir)
    
    # Show best problems
    if args.show_best > 0:
        print(f"\nTop {args.show_best} problems by avg reward:")
        print("-" * 70)
        best = find_best_chains(args.input_dir, args.show_best)
        for i, p in enumerate(best, 1):
            print(f"{i}. {p['id']}: reward={p['avg_reward']:.3f}, success={p['success_rate']:.1%}")
            print(f"   {p['problem'][:80]}...")
            print()
    
    # Show challenging problems
    if args.show_challenging > 0:
        print(f"\nTop {args.show_challenging} most challenging problems:")
        print("-" * 70)
        challenging = find_challenging_problems(args.input_dir, args.show_challenging)
        for i, p in enumerate(challenging, 1):
            print(f"{i}. {p['id']}: success={p['success_rate']:.1%}, reward={p['avg_reward']:.3f}")
            print(f"   {p['problem'][:80]}...")
            print()
    
    # Export for GRPO
    if args.export_grpo:
        num_exported = export_for_grpo(
            args.input_dir,
            args.export_file,
            args.min_reward
        )


if __name__ == "__main__":
    main()