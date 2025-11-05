#!/usr/bin/env python3
"""
Code Chain Expander
Hierarchical chain generation for coding problems with reward calculation.
"""
import ollama
from typing import List, Dict, Any, Optional
import json

from code_reward_metrics import CodeRewardCalculator


class CodeChainExecutor:
    """
    Executes hierarchical chains for coding problems with code-specific rewards.
    """
    
    def __init__(
        self,
        model: str = "qwen2.5-coder:7b",
        num_chains: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        reward_calculator: Optional[CodeRewardCalculator] = None,
        verbose: bool = False
    ):
        """
        Initialize code chain executor.
        
        Args:
            model: Ollama model name (qwen2.5-coder recommended for code)
            num_chains: Number of chains to generate
            temperature: Sampling temperature
            max_tokens: Max tokens per generation
            reward_calculator: Code reward calculator instance
            verbose: Print debug info
        """
        self.model = model
        self.num_chains = num_chains
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        
        # Initialize reward calculator
        if reward_calculator is None:
            self.reward_calculator = CodeRewardCalculator(
                use_codebleu=True,
                use_ast_similarity=True,
                use_codebert=True,
                success_threshold=0.7,
                verbose=verbose
            )
        else:
            self.reward_calculator = reward_calculator
    
    def _call_model(self, prompt: str) -> str:
        """Call Ollama model."""
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'temperature': self.temperature,
                    'num_predict': self.max_tokens
                }
            )
            return response['response'].strip()
        except Exception as e:
            if self.verbose:
                print(f"Model call error: {e}")
            return ""
    
    def _execute_subproblem_step(
        self,
        subproblem: Any,
        problem_context: str,
        dependency_results: Dict[str, str],
        wave_idx: int
    ) -> Dict[str, str]:
        """
        Execute a single subproblem step.
        
        Args:
            subproblem: Subproblem node
            problem_context: Original problem
            dependency_results: Results from dependencies
            wave_idx: Wave index
        
        Returns:
            Dictionary with reasoning and answer
        """
        # Build context from dependencies
        context_parts = []
        if dependency_results:
            context_parts.append("Previous steps:")
            for dep_id, result in dependency_results.items():
                context_parts.append(f"  {dep_id}: {result}")
        
        context = "\n".join(context_parts) if context_parts else "No previous steps."
        
        # Build prompt
        prompt = f"""You are solving a coding problem step by step.

Original Problem:
{problem_context}

Current Step Goal:
{subproblem.goal}

{context}

Provide your reasoning and solution for this step. If this is a code generation step, include the code in markdown code blocks.

Response:"""
        
        response = self._call_model(prompt)
        
        return {
            'reasoning': response,
            'answer': response
        }
    
    def _execute_single_chain(
        self,
        decomposition: Any,
        execution_waves: List[List[str]],
        problem: str,
        ground_truth: str,
        chain_index: int,
        language: str = "python"
    ) -> Dict[str, Any]:
        """Execute a single reasoning chain through all subproblems."""
        
        # Storage for intermediate results
        subproblem_results = {}
        execution_steps = []
        
        # Execute each wave in order
        for wave_idx, wave in enumerate(execution_waves):
            for subproblem_id in wave:
                subproblem = decomposition.get_node(subproblem_id)
                
                if subproblem is None:
                    continue
                
                # Get results from dependencies
                dependency_results = {}
                for dep_id in subproblem.depends_on:
                    if dep_id in subproblem_results:
                        dependency_results[dep_id] = subproblem_results[dep_id]
                
                # Execute this subproblem
                step_result = self._execute_subproblem_step(
                    subproblem=subproblem,
                    problem_context=problem,
                    dependency_results=dependency_results,
                    wave_idx=wave_idx
                )
                
                # Store result for future dependencies
                subproblem_results[subproblem_id] = step_result['answer']
                
                # Add to execution trace
                execution_steps.append({
                    'subproblem_id': subproblem_id,
                    'goal': subproblem.goal,
                    'reasoning': step_result['reasoning'],
                    'answer': step_result['answer'],
                    'wave': wave_idx,
                    'dependencies_used': list(dependency_results.keys())
                })
        
        # Build full reasoning text
        full_reasoning = self._build_full_reasoning_text(execution_steps)
        
        # Compute reward using code metrics
        reward_result = self.reward_calculator.compute_reward(
            predicted_code=full_reasoning,  # Will extract code internally
            reference_code=ground_truth,
            lang=language
        )
        
        return {
            'chain_id': f"chain_{chain_index}",
            'steps': execution_steps,
            'full_reasoning': full_reasoning,
            'final_reward': reward_result['final_reward'],
            'reward_details': reward_result,
            'num_steps': len(execution_steps),
            'temperature': self.temperature,
            'language': language
        }
    
    def _build_full_reasoning_text(self, execution_steps: List[Dict[str, Any]]) -> str:
        """Build complete reasoning chain as single text."""
        
        parts = []
        
        for i, step in enumerate(execution_steps, 1):
            parts.append(f"Step {i}: {step['goal']}")
            parts.append(f"{step['reasoning']}")
            parts.append("")  # Blank line
        
        return "\n".join(parts)
    
    def expand_chains(
        self,
        decomposition: Any,
        execution_waves: List[List[str]],
        problem: str,
        ground_truth: str,
        language: str = "python"
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple chains through the decomposition.
        
        Args:
            decomposition: Problem decomposition
            execution_waves: Execution order
            problem: Original problem
            ground_truth: Ground truth solution
            language: Programming language
        
        Returns:
            List of chain results
        """
        chains = []
        
        for i in range(self.num_chains):
            if self.verbose:
                print(f"  Generating chain {i+1}/{self.num_chains}...")
            
            chain = self._execute_single_chain(
                decomposition=decomposition,
                execution_waves=execution_waves,
                problem=problem,
                ground_truth=ground_truth,
                chain_index=i,
                language=language
            )
            
            chains.append(chain)
            
            if self.verbose:
                print(f"    Reward: {chain['final_reward']:.3f} "
                      f"({'✓ SUCCESS' if chain['reward_details']['is_successful'] else '✗ FAIL'})")
        
        return chains


def expand_code_decomposition(
    decomposition: Any,
    problem: str,
    ground_truth: str,
    executor: CodeChainExecutor,
    language: str = "python",
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Main function to expand a decomposition into multiple execution chains.
    
    Args:
        decomposition: Problem decomposition
        problem: Original problem statement
        ground_truth: Ground truth solution
        executor: CodeChainExecutor instance
        language: Programming language
        verbose: Print progress
    
    Returns:
        Dictionary containing chains and statistics
    """
    # Get execution waves
    execution_waves = decomposition.get_execution_waves()
    
    if verbose:
        print(f"\n  Execution plan: {len(execution_waves)} waves")
        for i, wave in enumerate(execution_waves):
            print(f"    Wave {i+1}: {len(wave)} subproblems")
    
    # Generate chains
    chains = executor.expand_chains(
        decomposition=decomposition,
        execution_waves=execution_waves,
        problem=problem,
        ground_truth=ground_truth,
        language=language
    )
    
    # Compute statistics
    total_chains = len(chains)
    successful_chains = sum(1 for c in chains if c['reward_details']['is_successful'])
    avg_reward = sum(c['final_reward'] for c in chains) / total_chains if total_chains > 0 else 0.0
    
    # Detailed reward statistics
    avg_codebleu = sum(c['reward_details']['codebleu'] for c in chains) / total_chains if total_chains > 0 else 0.0
    avg_ast = sum(c['reward_details']['ast_similarity'] for c in chains) / total_chains if total_chains > 0 else 0.0
    avg_codebert = sum(c['reward_details']['codebert_score'] for c in chains) / total_chains if total_chains > 0 else 0.0
    
    statistics = {
        'total_chains': total_chains,
        'successful_chains': successful_chains,
        'success_rate': successful_chains / total_chains if total_chains > 0 else 0.0,
        'avg_reward': avg_reward,
        'avg_codebleu': avg_codebleu,
        'avg_ast_similarity': avg_ast,
        'avg_codebert_score': avg_codebert,
        'min_reward': min((c['final_reward'] for c in chains), default=0.0),
        'max_reward': max((c['final_reward'] for c in chains), default=0.0)
    }
    
    return {
        'problem': problem,
        'ground_truth': ground_truth,
        'language': language,
        'decomposition': {
            'num_subproblems': len(decomposition.nodes),
            'num_waves': len(execution_waves)
        },
        'chains': chains,
        'statistics': statistics
    }