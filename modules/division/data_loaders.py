#!/usr/bin/env python3
"""Data loaders for various MATH datasets."""
import sys
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

# Add current directory to path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


class MATHDatasetLoader:
    """Base class for loading MATH datasets."""
    
    def __init__(self, dataset_name: str, split: str = "train"):
        self.dataset_name = dataset_name
        self.split = split
        self._dataset = None
    
    def load(self, max_problems: Optional[int] = None, category: Optional[str] = None) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Load problems from the dataset.
        
        Returns:
            List of (problem, answer, metadata) tuples
        """
        raise NotImplementedError
    
    def get_categories(self) -> List[str]:
        """Get list of available categories/types in the dataset."""
        raise NotImplementedError


class HendrycksCompetitionMathLoader(MATHDatasetLoader):
    """Loader for qwedsacf/competition_math (Hendrycks MATH dataset)."""
    
    def __init__(self, split: str = "train"):
        super().__init__("qwedsacf/competition_math", split)
        self._load_dataset()
    
    def _load_dataset(self):
        """Load the dataset from Hugging Face."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets package not found. Install with: pip install datasets --break-system-packages"
            )
        
        print(f"Loading {self.dataset_name} dataset...")
        self._dataset = load_dataset(self.dataset_name, split=self.split)
        print(f"✓ Loaded {len(self._dataset)} problems")
    
    def load(self, max_problems: Optional[int] = None, category: Optional[str] = None) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Load problems from Hendrycks MATH dataset.
        
        Args:
            max_problems: Maximum number of problems to load
            category: Filter by problem type (e.g., 'Algebra', 'Geometry')
        
        Returns:
            List of (problem, answer, metadata) tuples
        """
        if self._dataset is None:
            self._load_dataset()
        
        problems = []
        
        for i, item in enumerate(self._dataset):
            # Filter by category if specified
            if category and item.get('type') != category:
                continue
            
            problem = item['problem']
            solution = item['solution']
            
            # Extract answer from solution (enclosed in \boxed{})
            answer = self._extract_answer(solution)
            
            metadata = {
                'type': item.get('type', 'Unknown'),
                'level': item.get('level', 'Unknown'),
                'full_solution': solution,
                'dataset': self.dataset_name,
                'index': i
            }
            
            problems.append((problem, answer, metadata))
            
            if max_problems and len(problems) >= max_problems:
                break
        
        return problems
    
    def _extract_answer(self, solution: str) -> str:
        """Extract the final answer from a solution string."""
        # Look for \boxed{answer}
        import re
        match = re.search(r'\\boxed\{(.+?)\}', solution)
        if match:
            return match.group(1)
        
        # If no boxed answer found, return a placeholder
        return "[Answer not found in solution]"
    
    def get_categories(self) -> List[str]:
        """Get list of problem types in the dataset."""
        if self._dataset is None:
            self._load_dataset()
        
        types = set()
        for item in self._dataset:
            if 'type' in item:
                types.add(item['type'])
        
        return sorted(list(types))


class DeepMindMathLoader(MATHDatasetLoader):
    """Loader for DeepMind Mathematics dataset (backward compatibility)."""
    
    def __init__(self, split: str = "train"):
        super().__init__("deepmind/mathematics", split)
    
    def load(self, max_problems: Optional[int] = None, category: Optional[str] = None) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Load problems from local file or hardcoded examples."""
        # This maintains backward compatibility with the original batch_decompose.py
        examples_by_category = {
            "algebra__linear_1d": [
                ("Solve -42*r + 27*c = -1167 and 130*r + 4*c = 372 for r.", "4"),
                ("Solve 3*x + 7 = 22 for x.", "5"),
                ("What is the value of y in 2*y - 5 = 11?", "8"),
                ("Solve -5*a + 3 = -17 for a.", "4"),
                ("Find x when 4*x + 6 = 2*x + 14.", "4"),
            ],
        }
        
        category = category or "algebra__linear_1d"
        category_examples = examples_by_category.get(category, examples_by_category["algebra__linear_1d"])
        
        result = []
        idx = 0
        while len(result) < (max_problems or len(category_examples)):
            problem, answer = category_examples[idx % len(category_examples)]
            metadata = {
                'type': category,
                'level': 'Unknown',
                'dataset': 'deepmind/mathematics',
                'index': idx
            }
            result.append((problem, answer, metadata))
            idx += 1
            
            if idx >= (max_problems or len(category_examples)):
                break
        
        return result
    
    def get_categories(self) -> List[str]:
        """Get list of available categories."""
        return ["algebra__linear_1d"]


def create_loader(dataset_name: str, split: str = "train", config: Optional[str] = None) -> MATHDatasetLoader:
    """
    Factory function to create appropriate data loader.
    
    Args:
        dataset_name: Name of the dataset ('hendrycks_math', 'calc_svamp', 'calc_gsm8k', 'livemathbench', etc.)
        split: Dataset split (usually 'train', 'valid', 'test', or 'validation')
        config: Configuration for datasets that need it (e.g., LiveMathBench config)
    
    Returns:
        MATHDatasetLoader instance
    """
    dataset_name_lower = dataset_name.lower()
    
    if 'hendrycks' in dataset_name_lower or 'competition_math' in dataset_name_lower:
        return HendrycksCompetitionMathLoader(split)
    elif 'calc_svamp' in dataset_name_lower or 'svamp' in dataset_name_lower:
        # Calc-SVAMP only has 'test' split
        return CalcSVAMPLoader(split='test')
    elif 'calc_gsm8k' in dataset_name_lower or 'gsm8k' in dataset_name_lower:
        # Calc-GSM8K only has 'test' split
        return CalcGSM8KLoader(split='test')
    elif 'livemathbench' in dataset_name_lower or 'livemathbench' in dataset_name_lower:
        # LiveMathBench only has 'test' split
        config = config or "v202412_AMC_cn"
        return LiveMathBenchLoader(config, split='test')
    elif 'minif2f' in dataset_name_lower:
        return MiniF2FLoader(split)
    elif 'proofnet' in dataset_name_lower:
        # ProofNet typically uses 'validation' instead of 'train'
        if split == "train":
            split = "validation"
        return ProofNetLoader(split)
    elif 'deepmind' in dataset_name_lower:
        return DeepMindMathLoader(split)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported: 'hendrycks_math', 'calc_svamp', 'calc_gsm8k', 'livemathbench', 'minif2f', 'proofnet', 'deepmind'"
        )


class MiniF2FLoader(MATHDatasetLoader):
    """Loader for Tonic/MiniF2F dataset (Formal Mathematics)."""
    
    def __init__(self, split: str = "train"):
        super().__init__("Tonic/MiniF2F", split)
        self._load_dataset()
    
    def _load_dataset(self):
        """Load the dataset from Hugging Face."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets package not found. Install with: pip install datasets --break-system-packages"
            )
        
        print(f"Loading {self.dataset_name} dataset...")
        self._dataset = load_dataset(self.dataset_name, split=self.split)
        print(f"✓ Loaded {len(self._dataset)} problems")
    
    def load(self, max_problems: Optional[int] = None, category: Optional[str] = None) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Load problems from MiniF2F dataset.
        
        Args:
            max_problems: Maximum number of problems to load
            category: Filter by split type ('train', 'valid', 'test')
        
        Returns:
            List of (problem, answer, metadata) tuples
        """
        if self._dataset is None:
            self._load_dataset()
        
        problems = []
        
        for i, item in enumerate(self._dataset):
            # Filter by category (split type) if specified
            if category and item.get('split') != category:
                continue
            
            # Use informal_prefix as the problem statement
            problem = item.get('informal_prefix', item.get('name', 'No problem statement'))
            
            # The answer/goal is in the 'goal' field
            answer = item.get('goal', '[Goal not specified]')
            
            metadata = {
                'type': 'Formal Mathematics',
                'level': item.get('split', 'Unknown'),
                'name': item.get('name', f'problem_{i}'),
                'formal_statement': item.get('formal_statement', ''),
                'header': item.get('header', ''),
                'dataset': self.dataset_name,
                'index': i
            }
            
            problems.append((problem, answer, metadata))
            
            if max_problems and len(problems) >= max_problems:
                break
        
        return problems
    
    def get_categories(self) -> List[str]:
        """Get list of split types in the dataset."""
        if self._dataset is None:
            self._load_dataset()
        
        splits = set()
        for item in self._dataset:
            if 'split' in item:
                splits.add(item['split'])
        
        return sorted(list(splits))


class ProofNetLoader(MATHDatasetLoader):
    """Loader for hoskinson-center/proofnet dataset."""
    
    def __init__(self, split: str = "validation"):
        super().__init__("hoskinson-center/proofnet", split)
        self._load_dataset()
    
    def _load_dataset(self):
        """Load the dataset from Hugging Face."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets package not found. Install with: pip install datasets --break-system-packages"
            )
        
        print(f"Loading {self.dataset_name} dataset...")
        self._dataset = load_dataset(self.dataset_name, split=self.split)
        print(f"✓ Loaded {len(self._dataset)} problems")
    
    def load(self, max_problems: Optional[int] = None, category: Optional[str] = None) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Load problems from ProofNet dataset.
        
        Args:
            max_problems: Maximum number of problems to load
            category: Filter by problem source (e.g., 'Rudin', 'aime')
        
        Returns:
            List of (problem, answer, metadata) tuples
        """
        if self._dataset is None:
            self._load_dataset()
        
        problems = []
        
        for i, item in enumerate(self._dataset):
            # Extract category from problem ID (e.g., "Rudin|exercise_1_1a" -> "Rudin")
            problem_id = item.get('id', '')
            problem_category = problem_id.split('|')[0] if '|' in problem_id else 'Unknown'
            
            # Filter by category if specified
            if category and problem_category.lower() != category.lower():
                continue
            
            # Use nl_statement as the problem
            problem = item.get('nl_statement', 'No problem statement')
            
            # Use nl_proof as the answer/solution
            answer = item.get('nl_proof', '[Proof not available]')
            
            metadata = {
                'type': problem_category,
                'level': 'Undergraduate',
                'problem_id': problem_id,
                'formal_statement': item.get('formal_statement', ''),
                'dataset': self.dataset_name,
                'index': i
            }
            
            problems.append((problem, answer, metadata))
            
            if max_problems and len(problems) >= max_problems:
                break
        
        return problems
    
    def get_categories(self) -> List[str]:
        """Get list of problem sources in the dataset."""
        if self._dataset is None:
            self._load_dataset()
        
        categories = set()
        for item in self._dataset:
            problem_id = item.get('id', '')
            if '|' in problem_id:
                category = problem_id.split('|')[0]
                categories.add(category)
        
        return sorted(list(categories))


class CalcSVAMPLoader(MATHDatasetLoader):
    """Loader for MU-NLPC/Calc-svamp dataset."""
    
    def __init__(self, split: str = "test"):  # Changed from "train" to "test"
        super().__init__("MU-NLPC/Calc-svamp", split)
        self._load_dataset()
    
    def _load_dataset(self):
        """Load the dataset from Hugging Face."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets package not found. Install with: pip install datasets --break-system-packages"
            )
        
        print(f"Loading {self.dataset_name} dataset...")
        self._dataset = load_dataset(self.dataset_name, "default", split=self.split)
        print(f"✓ Loaded {len(self._dataset)} problems")
    
    def load(self, max_problems: Optional[int] = None, category: Optional[str] = None) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Load problems from Calc-SVAMP dataset.
        
        Args:
            max_problems: Maximum number of problems to load
            category: Not used for this dataset
        
        Returns:
            List of (problem, answer, metadata) tuples
        """
        if self._dataset is None:
            self._load_dataset()
        
        problems = []
        
        for i, item in enumerate(self._dataset):
            # Extract question and answer
            problem = item.get('question', item.get('Question', ''))
            answer = str(item.get('answer', item.get('Answer', '')))
            
            metadata = {
                'type': 'Arithmetic',
                'level': 'Elementary',
                'dataset': self.dataset_name,
                'index': i,
                'equation': item.get('equation', item.get('Equation', ''))
            }
            
            problems.append((problem, answer, metadata))
            
            if max_problems and len(problems) >= max_problems:
                break
        
        return problems
    
    def get_categories(self) -> List[str]:
        """Get list of categories."""
        return ["Arithmetic"]


class CalcGSM8KLoader(MATHDatasetLoader):
    """Loader for MU-NLPC/Calc-gsm8k dataset."""
    
    def __init__(self, split: str = "test"):  # Changed from "train" to "test"
        super().__init__("MU-NLPC/Calc-gsm8k", split)
        self._load_dataset()
    
    def _load_dataset(self):
        """Load the dataset from Hugging Face."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets package not found. Install with: pip install datasets --break-system-packages"
            )
        
        print(f"Loading {self.dataset_name} dataset...")
        self._dataset = load_dataset(self.dataset_name, "default", split=self.split)
        print(f"✓ Loaded {len(self._dataset)} problems")
    
    def load(self, max_problems: Optional[int] = None, category: Optional[str] = None) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Load problems from Calc-GSM8K dataset.
        
        Args:
            max_problems: Maximum number of problems to load
            category: Not used for this dataset
        
        Returns:
            List of (problem, answer, metadata) tuples
        """
        if self._dataset is None:
            self._load_dataset()
        
        problems = []
        
        for i, item in enumerate(self._dataset):
            # Extract question and answer
            problem = item.get('question', item.get('Question', ''))
            
            # Answer might be in 'answer' or need to be extracted from solution
            answer = item.get('answer', item.get('Answer', ''))
            if not answer and 'solution' in item:
                # Try to extract from solution (GSM8K format: "#### answer")
                solution = item['solution']
                if '####' in solution:
                    answer = solution.split('####')[-1].strip()
            
            answer = str(answer)
            
            metadata = {
                'type': 'Grade School Math',
                'level': 'Elementary',
                'dataset': self.dataset_name,
                'index': i
            }
            
            problems.append((problem, answer, metadata))
            
            if max_problems and len(problems) >= max_problems:
                break
        
        return problems
    
    def get_categories(self) -> List[str]:
        """Get list of categories."""
        return ["Grade School Math"]


class LiveMathBenchLoader(MATHDatasetLoader):
    """Loader for opencompass/LiveMathBench dataset."""
    
    def __init__(self, config: str = "v202412_AMC_cn", split: str = "test"):
        super().__init__(f"opencompass/LiveMathBench-{config}", split)
        self.config = config
        self._load_dataset()
    
    def _load_dataset(self):
        """Load the dataset from Hugging Face."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets package not found. Install with: pip install datasets --break-system-packages"
            )
        
        print(f"Loading opencompass/LiveMathBench ({self.config}) dataset...")
        self._dataset = load_dataset("opencompass/LiveMathBench", self.config, split=self.split)
        print(f"✓ Loaded {len(self._dataset)} problems")
    
    def load(self, max_problems: Optional[int] = None, category: Optional[str] = None) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Load problems from LiveMathBench dataset.
        
        Args:
            max_problems: Maximum number of problems to load
            category: Filter by problem type if available
        
        Returns:
            List of (problem, answer, metadata) tuples
        """
        if self._dataset is None:
            self._load_dataset()
        
        problems = []
        
        for i, item in enumerate(self._dataset):
            # Extract question and answer (field names may vary)
            problem = item.get('problem', item.get('question', item.get('Problem', '')))
            answer = str(item.get('answer', item.get('Answer', item.get('ground_truth', ''))))
            
            # Filter by category if specified
            item_category = item.get('category', item.get('type', 'Unknown'))
            if category and item_category != category:
                continue
            
            metadata = {
                'type': item_category,
                'level': item.get('difficulty', item.get('level', 'Unknown')),
                'dataset': self.dataset_name,
                'index': i,
                'config': self.config
            }
            
            # Add any other available fields
            if 'solution' in item:
                metadata['solution'] = item['solution']
            
            problems.append((problem, answer, metadata))
            
            if max_problems and len(problems) >= max_problems:
                break
        
        return problems
    
    def get_categories(self) -> List[str]:
        """Get list of problem categories."""
        if self._dataset is None:
            self._load_dataset()
        
        categories = set()
        for item in self._dataset:
            cat = item.get('category', item.get('type', 'Unknown'))
            categories.add(cat)
        
        return sorted(list(categories))


def get_available_datasets() -> List[str]:
    """Get list of available dataset loaders."""
    return [
        "hendrycks_math (qwedsacf/competition_math) - 12,500 competition problems",
        "calc_svamp (MU-NLPC/Calc-svamp) - Elementary arithmetic word problems",
        "calc_gsm8k (MU-NLPC/Calc-gsm8k) - Grade school math problems",
        "livemathbench (opencompass/LiveMathBench) - Live math competition problems",
        "minif2f (Tonic/MiniF2F) - 488 formal mathematics problems [PROOF DATASET]",
        "proofnet (hoskinson-center/proofnet) - 371 theorem proving problems [PROOF DATASET]",
        "deepmind (local/hardcoded) - Legacy examples",
    ]