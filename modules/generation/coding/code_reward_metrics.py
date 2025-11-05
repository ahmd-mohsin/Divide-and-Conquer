#!/usr/bin/env python3
"""
Code Reward Metrics
Computes rewards for code generation chains using multiple metrics:
- CodeBLEU (syntax + semantics)
- AST Edit Distance (structure)
- CodeBERTScore (semantic similarity)
"""
import ast
import numpy as np
from typing import Optional, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')


class CodeRewardCalculator:
    """
    Calculate rewards for code generation using multiple metrics.
    """
    
    def __init__(
        self,
        use_codebleu: bool = True,
        use_ast_similarity: bool = True,
        use_codebert: bool = True,
        success_threshold: float = 0.7,
        verbose: bool = False
    ):
        """
        Initialize code reward calculator.
        
        Args:
            use_codebleu: Use CodeBLEU metric
            use_ast_similarity: Use AST edit distance
            use_codebert: Use CodeBERTScore
            success_threshold: Threshold for marking chain as successful
            verbose: Print debug info
        """
        self.use_codebleu = use_codebleu
        self.use_ast_similarity = use_ast_similarity
        self.use_codebert = use_codebert
        self.success_threshold = success_threshold
        self.verbose = verbose
        
        # Lazy load models
        self._codebleu_scorer = None
        self._codebert_model = None
        self._codebert_tokenizer = None
        
        # Weights for combining metrics
        self.weights = {
            'codebleu': 0.4,
            'ast': 0.3,
            'codebert': 0.3
        }
    
    def _load_codebleu(self):
        """Lazy load CodeBLEU scorer."""
        if self._codebleu_scorer is None:
            try:
                from codebleu import calc_codebleu
                self._codebleu_scorer = calc_codebleu
                if self.verbose:
                    print("✓ CodeBLEU loaded")
            except ImportError:
                print("Warning: codebleu not installed. Install with: pip install codebleu")
                self.use_codebleu = False
        return self._codebleu_scorer
    
    def _load_codebert(self):
        """Lazy load CodeBERT model."""
        if self._codebert_model is None:
            try:
                from transformers import AutoTokenizer, AutoModel
                import torch
                
                model_name = "microsoft/codebert-base"
                self._codebert_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._codebert_model = AutoModel.from_pretrained(model_name)
                self._codebert_model.eval()
                
                if self.verbose:
                    print("✓ CodeBERT loaded")
            except ImportError:
                print("Warning: transformers not installed. Install with: pip install transformers torch")
                self.use_codebert = False
        return self._codebert_model, self._codebert_tokenizer
    
    def compute_codebleu(self, predicted: str, reference: str, lang: str = "python") -> float:
        """
        Compute CodeBLEU score.
        
        Args:
            predicted: Generated code
            reference: Ground truth code
            lang: Programming language
        
        Returns:
            CodeBLEU score (0.0 to 1.0)
        """
        if not self.use_codebleu:
            return 0.0
        
        try:
            calc_codebleu = self._load_codebleu()
            if calc_codebleu is None:
                return 0.0
            
            result = calc_codebleu(
                references=[[reference]],
                predictions=[predicted],
                lang=lang,
                weights=(0.25, 0.25, 0.25, 0.25),
                tokenizer=None
            )
            
            return result['codebleu']
        
        except Exception as e:
            if self.verbose:
                print(f"CodeBLEU error: {e}")
            return 0.0
    
    def compute_ast_similarity(self, predicted: str, reference: str) -> float:
        """
        Compute AST-based similarity using tree edit distance.
        
        Args:
            predicted: Generated code
            reference: Ground truth code
        
        Returns:
            AST similarity score (0.0 to 1.0)
        """
        if not self.use_ast_similarity:
            return 0.0
        
        try:
            # Parse both code snippets
            pred_tree = ast.parse(predicted)
            ref_tree = ast.parse(reference)
            
            # Extract AST node types
            pred_nodes = self._extract_ast_nodes(pred_tree)
            ref_nodes = self._extract_ast_nodes(ref_tree)
            
            # Compute similarity using Jaccard index
            if not pred_nodes and not ref_nodes:
                return 1.0
            
            intersection = len(pred_nodes & ref_nodes)
            union = len(pred_nodes | ref_nodes)
            
            similarity = intersection / union if union > 0 else 0.0
            
            return similarity
        
        except (SyntaxError, ValueError) as e:
            if self.verbose:
                print(f"AST parsing error: {e}")
            return 0.0
    
    def _extract_ast_nodes(self, tree) -> set:
        """Extract node types from AST."""
        nodes = set()
        for node in ast.walk(tree):
            nodes.add(type(node).__name__)
        return nodes
    
    def compute_codebert_score(self, predicted: str, reference: str) -> float:
        """
        Compute CodeBERT-based similarity score.
        
        Args:
            predicted: Generated code
            reference: Ground truth code
        
        Returns:
            CodeBERT similarity score (0.0 to 1.0)
        """
        if not self.use_codebert:
            return 0.0
        
        try:
            import torch
            
            model, tokenizer = self._load_codebert()
            if model is None:
                return 0.0
            
            # Tokenize
            pred_tokens = tokenizer(
                predicted,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            ref_tokens = tokenizer(
                reference,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            # Get embeddings
            with torch.no_grad():
                pred_outputs = model(**pred_tokens)
                ref_outputs = model(**ref_tokens)
            
            # Use [CLS] token embedding
            pred_embedding = pred_outputs.last_hidden_state[:, 0, :].squeeze()
            ref_embedding = ref_outputs.last_hidden_state[:, 0, :].squeeze()
            
            # Cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                pred_embedding.unsqueeze(0),
                ref_embedding.unsqueeze(0)
            ).item()
            
            # Convert from [-1, 1] to [0, 1]
            similarity = (similarity + 1) / 2
            
            return similarity
        
        except Exception as e:
            if self.verbose:
                print(f"CodeBERT error: {e}")
            return 0.0
    
    def compute_reward(
        self,
        predicted_code: str,
        reference_code: str,
        lang: str = "python"
    ) -> Dict[str, Any]:
        """
        Compute combined reward using all enabled metrics.
        
        Args:
            predicted_code: Generated code
            reference_code: Ground truth code
            lang: Programming language
        
        Returns:
            Dictionary with individual scores and final reward
        """
        scores = {}
        
        # Compute individual metrics
        if self.use_codebleu:
            scores['codebleu'] = self.compute_codebleu(predicted_code, reference_code, lang)
        else:
            scores['codebleu'] = 0.0
        
        if self.use_ast_similarity:
            scores['ast_similarity'] = self.compute_ast_similarity(predicted_code, reference_code)
        else:
            scores['ast_similarity'] = 0.0
        
        if self.use_codebert:
            scores['codebert_score'] = self.compute_codebert_score(predicted_code, reference_code)
        else:
            scores['codebert_score'] = 0.0
        
        # Compute weighted combination
        active_metrics = sum([self.use_codebleu, self.use_ast_similarity, self.use_codebert])
        
        if active_metrics == 0:
            final_reward = 0.0
        else:
            # Normalize weights based on active metrics
            total_weight = sum([
                self.weights['codebleu'] if self.use_codebleu else 0,
                self.weights['ast'] if self.use_ast_similarity else 0,
                self.weights['codebert'] if self.use_codebert else 0
            ])
            
            final_reward = (
                scores['codebleu'] * self.weights['codebleu'] +
                scores['ast_similarity'] * self.weights['ast'] +
                scores['codebert_score'] * self.weights['codebert']
            ) / total_weight
        
        # Check if successful
        is_successful = final_reward >= self.success_threshold
        
        return {
            'codebleu': scores['codebleu'],
            'ast_similarity': scores['ast_similarity'],
            'codebert_score': scores['codebert_score'],
            'final_reward': final_reward,
            'is_successful': is_successful,
            'threshold': self.success_threshold
        }
    
    def compute_chain_reward(
        self,
        generated_chain: str,
        reference_solution: str,
        lang: str = "python"
    ) -> float:
        """
        Compute reward for entire reasoning chain ending with code.
        
        Args:
            generated_chain: Full chain including reasoning + code
            reference_solution: Ground truth solution code
            lang: Programming language
        
        Returns:
            Final reward score (0.0 to 1.0)
        """
        # Extract code from chain (assume code is in last step or marked)
        generated_code = self._extract_code_from_chain(generated_chain)
        
        # Compute reward
        result = self.compute_reward(generated_code, reference_solution, lang)
        
        return result['final_reward']
    
    def _extract_code_from_chain(self, chain: str) -> str:
        """
        Extract code from reasoning chain.
        Assumes code is in markdown code blocks or at the end.
        """
        # Try to find code in markdown blocks
        import re
        
        # Pattern for ```python ... ```
        pattern = r'```(?:python)?\s*(.*?)```'
        matches = re.findall(pattern, chain, re.DOTALL)
        
        if matches:
            # Return last code block
            return matches[-1].strip()
        
        # If no code blocks, assume entire chain is code
        # (fallback - may need adjustment based on your format)
        return chain.strip()


def test_metrics():
    """Test the reward calculator."""
    
    calculator = CodeRewardCalculator(verbose=True)
    
    reference = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
    
    # Good prediction (similar logic)
    good_pred = """
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)
"""
    
    # Bad prediction (different logic)
    bad_pred = """
def fibonacci(n):
    result = 0
    for i in range(n):
        result += i
    return result
"""
    
    print("\nTesting GOOD prediction:")
    result_good = calculator.compute_reward(good_pred, reference)
    print(f"  CodeBLEU: {result_good['codebleu']:.3f}")
    print(f"  AST Similarity: {result_good['ast_similarity']:.3f}")
    print(f"  CodeBERT: {result_good['codebert_score']:.3f}")
    print(f"  Final Reward: {result_good['final_reward']:.3f}")
    print(f"  Success: {result_good['is_successful']}")
    
    print("\nTesting BAD prediction:")
    result_bad = calculator.compute_reward(bad_pred, reference)
    print(f"  CodeBLEU: {result_bad['codebleu']:.3f}")
    print(f"  AST Similarity: {result_bad['ast_similarity']:.3f}")
    print(f"  CodeBERT: {result_bad['codebert_score']:.3f}")
    print(f"  Final Reward: {result_bad['final_reward']:.3f}")
    print(f"  Success: {result_bad['is_successful']}")


if __name__ == "__main__":
    test_metrics()