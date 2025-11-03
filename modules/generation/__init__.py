"""HCOT Module 2: Chain-of-Thought Generation

This module generates multiple diverse reasoning chains for each subproblem
in a decomposed math problem.

Example usage:
    from generation.chain_expander import ChainExpander
    from generation.batch_chain_generation import batch_generate_chains
    
    # Generate chains for a dataset
    stats = batch_generate_chains(
        dataset_name="hendrycks_math",
        num_problems=10,
        num_chains=5
    )
"""

__version__ = "0.1.0"

from .chain_expander import ChainExpander, expand_decomposition
from .batch_chain_generation import batch_generate_chains, ChainDataset

__all__ = [
    "ChainExpander",
    "expand_decomposition",
    "batch_generate_chains",
    "ChainDataset"
]