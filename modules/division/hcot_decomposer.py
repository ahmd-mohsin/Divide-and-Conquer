# hcot_decomposer.py
"""Hierarchical Chain-of-Thought decomposition system."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

from pydantic import ValidationError

from config import HCOTConfig, ModelConfig, DecompositionConfig
from schemas import Decomposition, DECOMPOSITION_SCHEMA
from llm_clients import create_client, LLMClient


class HierarchicalDecomposer:
    """
    Decomposes complex math problems into hierarchical sub-problems.
    
    This is Module 1 of the HCOT system. Future modules:
    - Module 2: CoT expansion for each sub-problem
    - Module 3: PRM scoring and pruning
    - Module 4: RL agent training
    - Module 5: Solution merging
    """
    
    def __init__(
        self,
        config: HCOTConfig,
        client: Optional[LLMClient] = None
    ):
        """
        Initialize the decomposer.
        
        Args:
            config: HCOT configuration
            client: Optional pre-configured LLM client (auto-created if None)
        """
        self.config = config
        self.client = client or create_client(config.model)
        
        # Load prompts
        self._load_prompts()
    
    def _load_prompts(self):
        """Load system and user prompt templates."""
        prompts_path = Path(self.config.prompts_path)
        if not prompts_path.exists():
            raise FileNotFoundError(f"Prompts file not found: {prompts_path}")
        
        with open(prompts_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.system_prompt = data.get("system_prompt", "")
        self.user_template = data.get("user_template", "")
        
        if not self.system_prompt or not self.user_template:
            raise ValueError("Prompts file must contain 'system_prompt' and 'user_template'")
    
    def decompose(
        self,
        problem: str,
        depth: Optional[int] = None,
        branching: Optional[int] = None,
        retry_on_failure: bool = True
    ) -> Decomposition:
        """
        Decompose a problem into hierarchical sub-problems.
        
        Args:
            problem: The mathematical problem to decompose
            depth: Maximum depth of hierarchy (uses config default if None)
            branching: Maximum children per node (uses config default if None)
            retry_on_failure: Whether to retry on validation failure
            
        Returns:
            Decomposition object with validated hierarchy
            
        Raises:
            RuntimeError: If decomposition fails after retries
        """
        depth = depth or self.config.decomposition.depth_limit
        branching = branching or self.config.decomposition.branching_limit
        
        for attempt in range(self.config.decomposition.max_retries):
            try:
                if self.config.verbose:
                    print(f"Decomposition attempt {attempt + 1}/{self.config.decomposition.max_retries}")
                
                decomp = self._decompose_once(problem, depth, branching)
                
                # Validate hierarchy structure
                is_valid, errors = decomp.validate_hierarchy()
                if not is_valid:
                    if self.config.verbose:
                        print(f"Validation errors: {errors}")
                    if retry_on_failure and attempt < self.config.decomposition.max_retries - 1:
                        continue
                    raise RuntimeError(f"Invalid hierarchy structure: {errors}")
                
                if self.config.verbose:
                    print(f"Successfully decomposed into {len(decomp.nodes)} sub-problems")
                
                return decomp
                
            except (json.JSONDecodeError, ValidationError) as e:
                if self.config.verbose:
                    print(f"Attempt {attempt + 1} failed: {e}")
                
                if not retry_on_failure or attempt >= self.config.decomposition.max_retries - 1:
                    raise RuntimeError(f"Decomposition failed after {attempt + 1} attempts: {e}")
        
        raise RuntimeError("Decomposition failed: max retries exceeded")
    
    def _decompose_once(self, problem: str, depth: int, branching: int) -> Decomposition:
        """Perform a single decomposition attempt."""
        # Format user prompt
        user_prompt = self.user_template.format(
            problem=problem.strip(),
            depth=depth,
            branching=branching
        )
        
        # Get JSON response from LLM
        raw_json = self.client.chat_json(
            system=self.system_prompt,
            user=user_prompt,
            schema=DECOMPOSITION_SCHEMA
        )
        
        # Parse JSON
        data = self._parse_json_response(raw_json)
        
        # Validate with Pydantic
        decomp = Decomposition(**data)
        
        return decomp
    
    def _parse_json_response(self, raw: str) -> dict:
        """
        Parse JSON from LLM response, handling markdown code blocks.
        
        Args:
            raw: Raw response string
            
        Returns:
            Parsed JSON dict
        """
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try cleaning markdown code blocks
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                # Remove markdown code fences
                lines = cleaned.split("\n")
                # Remove first line (```json or ```)
                lines = lines[1:]
                # Remove last line if it's ```
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned = "\n".join(lines)
            
            return json.loads(cleaned)
    
    def decompose_batch(
        self,
        problems: list[str],
        **kwargs
    ) -> list[Decomposition]:
        """
        Decompose multiple problems.
        
        Args:
            problems: List of problems to decompose
            **kwargs: Arguments passed to decompose()
            
        Returns:
            List of Decomposition objects
        """
        results = []
        for i, problem in enumerate(problems):
            if self.config.verbose:
                print(f"\n=== Problem {i+1}/{len(problems)} ===")
            
            try:
                decomp = self.decompose(problem, **kwargs)
                results.append(decomp)
            except Exception as e:
                if self.config.verbose:
                    print(f"Failed to decompose problem {i+1}: {e}")
                results.append(None)
        
        return results
    
    def visualize_hierarchy(self, decomp: Decomposition) -> str:
        """
        Create a text visualization of the hierarchy.
        
        Args:
            decomp: Decomposition to visualize
            
        Returns:
            String representation of the hierarchy tree
        """
        lines = []
        lines.append(f"Problem: {decomp.problem[:80]}...")
        lines.append(f"Nodes: {len(decomp.nodes)} | Depth: {decomp.depth_limit} | Branching: {decomp.branching_limit}")
        lines.append("")
        
        # Build tree recursively
        def add_node(node_id: str, indent: int = 0):
            node = decomp.get_node(node_id)
            if not node:
                return
            
            prefix = "  " * indent + "├─ " if indent > 0 else ""
            lines.append(f"{prefix}[{node.id}] {node.goal[:60]}")
            
            if node.depends_on:
                dep_str = ", ".join(node.depends_on)
                lines.append(f"{'  ' * (indent + 1)}↳ depends: {dep_str}")
            
            if node.suggested_check.value != "none":
                lines.append(f"{'  ' * (indent + 1)}✓ check: {node.suggested_check.value}")
            
            # Recurse to children
            for child in decomp.get_children(node.id):
                add_node(child.id, indent + 1)
        
        # Start with root nodes
        for root in decomp.get_root_nodes():
            add_node(root.id)
        
        return "\n".join(lines)


def quick_decompose(
    problem: str,
    model: str = "llama3.1:8b",
    prompts_path: str = "hcot.json",
    depth: int = 3,
    branching: int = 3,
    verbose: bool = False,
    **model_kwargs
) -> Decomposition:
    """
    Quick helper function to decompose a single problem.
    
    Args:
        problem: Problem to decompose
        model: Model identifier (e.g., "llama3.1:8b", "gpt-4")
        prompts_path: Path to prompts JSON file
        depth: Maximum hierarchy depth
        branching: Maximum children per node
        verbose: Print debug information
        **model_kwargs: Additional model configuration
        
    Returns:
        Decomposition object
    """
    from config import get_model_config
    
    model_config = get_model_config(model, **model_kwargs)
    decomp_config = DecompositionConfig(
        depth_limit=depth,
        branching_limit=branching
    )
    
    config = HCOTConfig(
        model=model_config,
        decomposition=decomp_config,
        prompts_path=prompts_path,
        verbose=verbose
    )
    
    decomposer = HierarchicalDecomposer(config)
    return decomposer.decompose(problem)