#!/usr/bin/env python3
"""
Code Dataset Loaders
Loaders for APPS, DS-1000, and Codeforces datasets.
"""
import json
from typing import List, Tuple, Dict, Any, Optional
from abc import ABC, abstractmethod


class CodeDatasetLoader(ABC):
    """Base class for code dataset loaders."""
    
    def __init__(self, dataset_name: str, split: str = "train"):
        self.dataset_name = dataset_name
        self.split = split
        self._dataset = None
    
    @abstractmethod
    def load(self, max_problems: Optional[int] = None) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Load problems from dataset.
        
        Returns:
            List of (problem, solution, metadata) tuples
        """
        pass
    
    @abstractmethod
    def _load_dataset(self):
        """Load the dataset from source."""
        pass


class APPSLoader(CodeDatasetLoader):
    """Loader for APPS (Automated Programming Progress Standard) dataset."""
    
    def __init__(self, split: str = "train"):
        super().__init__("codeparrot/apps", split)
        self._load_dataset()
    
    def _load_dataset(self):
        """Load APPS dataset from Hugging Face."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets package not found. Install with: pip install datasets"
            )
        
        print(f"Loading {self.dataset_name} dataset (split: {self.split})...")
        self._dataset = load_dataset(self.dataset_name, split=self.split)
        print(f"✓ Loaded {len(self._dataset)} problems")
    
    def load(
        self,
        max_problems: Optional[int] = None,
        difficulty: Optional[str] = None
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Load problems from APPS dataset.
        
        Args:
            max_problems: Maximum number of problems to load
            difficulty: Filter by difficulty ('introductory', 'interview', 'competition')
        
        Returns:
            List of (problem, solution, metadata) tuples
        """
        if self._dataset is None:
            self._load_dataset()
        
        problems = []
        
        for i, item in enumerate(self._dataset):
            # Filter by difficulty if specified
            if difficulty and item.get('difficulty') != difficulty:
                continue
            
            # Parse solutions from JSON string
            try:
                solutions = json.loads(item['solutions']) if item['solutions'] else []
            except (json.JSONDecodeError, TypeError):
                solutions = []
            
            # Skip if no solutions
            if not solutions:
                continue
            
            # Use first solution as ground truth
            solution = solutions[0]
            
            # Parse input/output test cases
            try:
                input_output = json.loads(item['input_output']) if item['input_output'] else {}
            except (json.JSONDecodeError, TypeError):
                input_output = {}
            
            metadata = {
                'type': 'Coding',
                'dataset': self.dataset_name,
                'problem_id': item.get('problem_id', i),
                'difficulty': item.get('difficulty', 'unknown'),
                'url': item.get('url', ''),
                'starter_code': item.get('starter_code', ''),
                'input_output': input_output,
                'all_solutions': solutions,
                'language': 'python',
                'index': i
            }
            
            problems.append((item['question'], solution, metadata))
            
            if max_problems and len(problems) >= max_problems:
                break
        
        return problems
    
    def get_difficulties(self) -> List[str]:
        """Get list of difficulty levels."""
        return ['introductory', 'interview', 'competition']


class DS1000Loader(CodeDatasetLoader):
    """Loader for DS-1000 (Data Science) dataset."""
    
    def __init__(self, split: str = "test"):
        super().__init__("xlangai/DS-1000", split)
        self._load_dataset()
    
    def _load_dataset(self):
        """Load DS-1000 dataset from Hugging Face."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets package not found. Install with: pip install datasets"
            )
        
        print(f"Loading {self.dataset_name} dataset...")
        # DS-1000 may not have explicit splits, load full dataset
        try:
            self._dataset = load_dataset(self.dataset_name, split=self.split)
        except ValueError:
            # If split doesn't exist, load full dataset
            self._dataset = load_dataset(self.dataset_name)
            if hasattr(self._dataset, 'keys'):
                # Pick first available split
                first_split = list(self._dataset.keys())[0]
                self._dataset = self._dataset[first_split]
        
        print(f"✓ Loaded {len(self._dataset)} problems")
    
    def load(
        self,
        max_problems: Optional[int] = None,
        library: Optional[str] = None
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Load problems from DS-1000 dataset.
        
        Args:
            max_problems: Maximum number of problems to load
            library: Filter by library (e.g., 'Pandas', 'NumPy', 'PyTorch')
        
        Returns:
            List of (problem, solution, metadata) tuples
        """
        if self._dataset is None:
            self._load_dataset()
        
        problems = []
        
        for i, item in enumerate(self._dataset):
            # Filter by library if specified
            if library and item.get('lib', '').lower() != library.lower():
                continue
            
            # Extract problem and solution
            problem = item.get('prompt', item.get('question', ''))
            solution = item.get('reference_code', item.get('canonical_solution', ''))
            
            # Skip if no solution
            if not solution:
                continue
            
            metadata = {
                'type': 'Data Science',
                'dataset': self.dataset_name,
                'library': item.get('lib', 'unknown'),
                'difficulty': item.get('difficulty', 'unknown'),
                'language': 'python',
                'index': i
            }
            
            # Add test cases if available
            if 'test' in item:
                metadata['test_cases'] = item['test']
            
            problems.append((problem, solution, metadata))
            
            if max_problems and len(problems) >= max_problems:
                break
        
        return problems


class CodeforcesLoader(CodeDatasetLoader):
    """Loader for Codeforces dataset."""
    
    def __init__(self, split: str = "train"):
        super().__init__("open-r1/codeforces", split)
        self._load_dataset()
    
    def _load_dataset(self):
        """Load Codeforces dataset from Hugging Face."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets package not found. Install with: pip install datasets"
            )
        
        print(f"Loading {self.dataset_name} dataset (split: {self.split})...")
        try:
            self._dataset = load_dataset(self.dataset_name, "default", split=self.split)
        except Exception as e:
            print(f"Note: {e}")
            # Try without config
            self._dataset = load_dataset(self.dataset_name, split=self.split)
        
        print(f"✓ Loaded {len(self._dataset)} problems")
    
    def load(
        self,
        max_problems: Optional[int] = None,
        rating: Optional[int] = None
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Load problems from Codeforces dataset.
        
        Args:
            max_problems: Maximum number of problems to load
            rating: Filter by minimum rating
        
        Returns:
            List of (problem, solution, metadata) tuples
        """
        if self._dataset is None:
            self._load_dataset()
        
        problems = []
        
        for i, item in enumerate(self._dataset):
            # Filter by rating if specified
            problem_rating = item.get('rating', 0)
            if rating and problem_rating < rating:
                continue
            
            # Extract problem and solution
            problem = item.get('problem', item.get('description', item.get('question', '')))
            solution = item.get('solution', item.get('code', ''))
            
            # Skip if no solution
            if not solution:
                continue
            
            metadata = {
                'type': 'Competitive Programming',
                'dataset': self.dataset_name,
                'rating': problem_rating,
                'tags': item.get('tags', []),
                'language': item.get('language', 'python'),
                'index': i
            }
            
            # Add test cases if available
            if 'tests' in item:
                metadata['test_cases'] = item['tests']
            if 'input' in item and 'output' in item:
                metadata['test_cases'] = {
                    'inputs': [item['input']],
                    'outputs': [item['output']]
                }
            
            problems.append((problem, solution, metadata))
            
            if max_problems and len(problems) >= max_problems:
                break
        
        return problems


def create_code_loader(
    dataset_name: str,
    split: str = "train"
) -> CodeDatasetLoader:
    """
    Factory function to create appropriate code data loader.
    
    Args:
        dataset_name: Name of dataset ('apps', 'ds1000', 'codeforces')
        split: Dataset split
    
    Returns:
        CodeDatasetLoader instance
    """
    dataset_name_lower = dataset_name.lower()
    
    if 'apps' in dataset_name_lower:
        return APPSLoader(split)
    elif 'ds' in dataset_name_lower or 'ds1000' in dataset_name_lower or 'ds-1000' in dataset_name_lower:
        # DS-1000 typically uses test split
        return DS1000Loader(split='test')
    elif 'codeforces' in dataset_name_lower:
        return CodeforcesLoader(split)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported: 'apps', 'ds1000', 'codeforces'"
        )


def get_available_code_datasets() -> List[str]:
    """Get list of available code dataset loaders."""
    return [
        "apps (codeparrot/apps) - 10,000 Python programming problems",
        "ds1000 (xlangai/DS-1000) - Data science problems (Pandas, NumPy, etc.)",
        "codeforces (open-r1/codeforces) - Competitive programming problems",
    ]


if __name__ == "__main__":
    # Test loaders
    print("Testing APPS loader...")
    apps = APPSLoader(split="train")
    problems = apps.load(max_problems=5)
    print(f"Loaded {len(problems)} APPS problems")
    
    print("\nTesting DS-1000 loader...")
    ds1000 = DS1000Loader()
    problems = ds1000.load(max_problems=5)
    print(f"Loaded {len(problems)} DS-1000 problems")
    
    print("\nTesting Codeforces loader...")
    cf = CodeforcesLoader(split="train")
    problems = cf.load(max_problems=5)
    print(f"Loaded {len(problems)} Codeforces problems")