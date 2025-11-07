"""
Math Dataset Loaders
Load math problems from various datasets with proper ground truth extraction.
"""
import re
from typing import List, Tuple, Dict, Optional


def extract_numerical_answer(answer_text: str) -> str:
    """
    Extract numerical answer from text with various formats.
    
    Handles:
    - GSM8K format: "#### 42"
    - Already numeric: "42" or "42.5"
    - Embedded in text: "The answer is 42"
    
    Args:
        answer_text: Raw answer text
    
    Returns:
        Extracted numerical answer as string
    """
    if not answer_text or not isinstance(answer_text, str):
        return str(answer_text) if answer_text else ""
    
    answer_text = answer_text.strip()
    
    # Empty string check
    if not answer_text:
        return ""
    
    # Method 1: GSM8K format with ####
    if '####' in answer_text:
        answer = answer_text.split('####')[-1].strip()
        # Clean any remaining newlines
        answer = answer.split('\n')[0].strip()
        # Remove any remaining whitespace or special chars
        answer = answer.replace(',', '').strip()
        return answer
    
    # Method 2: Already just a number
    try:
        float(answer_text.replace(',', ''))
        return answer_text.replace(',', '')
    except ValueError:
        pass
    
    # Method 3: Extract last number from text
    # Remove commas first
    clean_text = answer_text.replace(',', '')
    numbers = re.findall(r'-?\d+\.?\d*', clean_text)
    if numbers:
        return numbers[-1]
    
    # Method 4: Return original if no number found
    return answer_text


class MathDataLoader:
    """Base class for math dataset loaders"""
    
    def load(self, split: str = "train", max_problems: Optional[int] = None) -> List[Tuple[str, str, Dict]]:
        """
        Load problems from dataset.
        
        Returns:
            List of (problem, ground_truth, metadata) tuples
        """
        raise NotImplementedError


class GSM8KLoader(MathDataLoader):
    """Loader for GSM8K dataset"""
    
    def load(self, split: str = "train", max_problems: Optional[int] = None) -> List[Tuple[str, str, Dict]]:
        """Load GSM8K problems with proper ground truth extraction"""
        from datasets import load_dataset
        
        print(f"Loading GSM8K dataset (split: {split})...")
        dataset = load_dataset("openai/gsm8k", "main", split=split)
        
        if max_problems:
            dataset = dataset.select(range(min(max_problems, len(dataset))))
        
        problems = []
        for idx, item in enumerate(dataset):
            problem = item['question']
            answer_text = item['answer']
            
            # Extract numerical answer
            ground_truth = extract_numerical_answer(answer_text)
            
            # Verify we got a valid answer
            if not ground_truth:
                print(f"⚠ Warning: Empty ground truth for problem {idx}")
                print(f"  Original answer: {answer_text[:100]}")
                ground_truth = "0"  # Fallback
            
            metadata = {
                'dataset': 'gsm8k',
                'type': 'word problem',
                'difficulty': 'grade school',
                'full_solution': answer_text,
                'index': idx
            }
            
            problems.append((problem, ground_truth, metadata))
        
        print(f"✓ Loaded {len(problems)} problems")
        return problems


class CalcSVAMPLoader(MathDataLoader):
    """Loader for Calc-SVAMP dataset"""
    
    def load(self, split: str = "train", max_problems: Optional[int] = None) -> List[Tuple[str, str, Dict]]:
        """Load Calc-SVAMP problems with proper answer extraction"""
        from datasets import load_dataset
        
        print(f"Loading Calc-SVAMP dataset...")
        
        # Try different dataset variations
        dataset = None
        dataset_names = [
            ("MU-NLPC/Calc-gsm8k", "train"),
            ("MU-NLPC/Calc-X", "train"),
            ("MU-NLPC/Calc-ape210k", "train")
        ]
        
        for ds_name, ds_split in dataset_names:
            try:
                print(f"  Trying {ds_name}...")
                dataset = load_dataset(ds_name, split=ds_split)
                print(f"  ✓ Loaded from {ds_name}")
                break
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                continue
        
        if dataset is None:
            raise RuntimeError("Could not load Calc-SVAMP dataset from any source")
        
        if max_problems:
            dataset = dataset.select(range(min(max_problems, len(dataset))))
        
        problems = []
        for idx, item in enumerate(dataset):
            problem = item.get('question', item.get('body', ''))
            answer = item.get('answer', item.get('solution', ''))
            
            # Extract numerical answer
            if isinstance(answer, (int, float)):
                ground_truth = str(answer)
            else:
                ground_truth = extract_numerical_answer(str(answer))
            
            # Verify we got a valid answer
            if not ground_truth:
                print(f"⚠ Warning: Empty ground truth for problem {idx}")
                print(f"  Original answer: {answer}")
                ground_truth = "0"  # Fallback
            
            metadata = {
                'dataset': 'calc-svamp',
                'type': 'word problem',
                'difficulty': 'elementary',
                'equation': item.get('equation', ''),
                'full_answer': str(answer),
                'index': idx
            }
            
            problems.append((problem, ground_truth, metadata))
        
        print(f"✓ Loaded {len(problems)} problems")
        return problems


class HendrycksLoader(MathDataLoader):
    """Loader for Hendrycks MATH dataset"""
    
    def load(self, split: str = "train", max_problems: Optional[int] = None) -> List[Tuple[str, str, Dict]]:
        """Load Hendrycks MATH problems"""
        from datasets import load_dataset
        
        print(f"Loading Hendrycks MATH dataset (split: {split})...")
        dataset = load_dataset("lighteval/MATH", split=split)
        
        if max_problems:
            dataset = dataset.select(range(min(max_problems, len(dataset))))
        
        problems = []
        for idx, item in enumerate(dataset):
            problem = item['problem']
            solution = item['solution']
            
            # Try to extract boxed answer
            ground_truth = solution
            if '\\boxed{' in solution:
                match = re.search(r'\\boxed\{([^}]+)\}', solution)
                if match:
                    ground_truth = match.group(1)
            
            metadata = {
                'dataset': 'hendrycks',
                'type': item.get('type', 'unknown'),
                'level': item.get('level', 'unknown'),
                'full_solution': solution,
                'index': idx
            }
            
            problems.append((problem, ground_truth, metadata))
        
        print(f"✓ Loaded {len(problems)} problems")
        return problems


def create_math_loader(dataset_name: str) -> MathDataLoader:
    """
    Factory function to create appropriate loader.
    
    Args:
        dataset_name: 'gsm8k', 'calc-svamp', 'hendrycks', etc.
    
    Returns:
        Appropriate MathDataLoader instance
    """
    loaders = {
        'gsm8k': GSM8KLoader,
        'calc-svamp': CalcSVAMPLoader,
        'calc': CalcSVAMPLoader,
        'hendrycks': HendrycksLoader,
        'hendrycks-math': HendrycksLoader,
    }
    
    dataset_name = dataset_name.lower()
    if dataset_name not in loaders:
        raise ValueError(
            f"Unknown dataset: {dataset_name}\n"
            f"Available: {', '.join(loaders.keys())}"
        )
    
    return loaders[dataset_name]()


# Test function
if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING MATH DATA LOADERS")
    print("="*70)
    
    for dataset_name in ['gsm8k', 'calc-svamp']:
        print(f"\n--- Testing {dataset_name} ---")
        try:
            loader = create_math_loader(dataset_name)
            problems = loader.load(max_problems=3)
            
            for i, (problem, ground_truth, metadata) in enumerate(problems, 1):
                print(f"\nProblem {i}:")
                print(f"  Question: {problem[:80]}...")
                print(f"  Ground Truth: '{ground_truth}'")
                print(f"  Metadata: {metadata}")
                
                if not ground_truth or ground_truth == "":
                    print(f"  ⚠ WARNING: Empty ground truth!")
                else:
                    print(f"  ✓ Ground truth OK")
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)