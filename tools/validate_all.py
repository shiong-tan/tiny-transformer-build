"""
Validation Script for Tiny Transformer Course

Validates all code and notebooks to ensure they work correctly.
Inspired by yhilpisch/llmcode validation patterns.

Features:
- Run all unit tests
- Execute notebooks to ensure no errors
- Report timing and success/failure
- Structured output for CI/CD

Usage:
    python tools/validate_all.py                 # Validate everything
    python tools/validate_all.py --tests-only    # Only run tests
    python tools/validate_all.py --notebooks-only # Only validate notebooks
"""

import sys
import time
import subprocess
from pathlib import Path
from typing import List, Tuple
import argparse


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str) -> None:
    """Print formatted section header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}  {text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.END}\n")


def print_result(name: str, passed: bool, duration: float, details: str = "") -> None:
    """Print result of a validation check."""
    status = f"{Colors.GREEN}✓ PASS{Colors.END}" if passed else f"{Colors.RED}✗ FAIL{Colors.END}"
    time_str = f"({duration:.2f}s)"
    
    print(f"{status} {name:<50s} {time_str}")
    
    if details and not passed:
        print(f"  {Colors.YELLOW}→ {details}{Colors.END}")


def run_pytest() -> Tuple[bool, float, str]:
    """
    Run pytest on all test files.
    
    Returns:
        (passed, duration, error_message)
    """
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ['python', '-m', 'pytest', '--tb=short', '-v'],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        duration = time.time() - start_time
        passed = result.returncode == 0
        
        error_msg = result.stdout if not passed else ""
        
        return passed, duration, error_msg
        
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        return False, duration, "Tests timed out after 5 minutes"
    except Exception as e:
        duration = time.time() - start_time
        return False, duration, str(e)


def run_module_tests(module_path: Path) -> Tuple[bool, float, str]:
    """
    Run tests for a specific module.
    
    Args:
        module_path: Path to module directory
        
    Returns:
        (passed, duration, error_message)
    """
    test_dir = module_path / 'tests'
    
    if not test_dir.exists():
        return True, 0.0, "No tests found (skipped)"
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            ['python', '-m', 'pytest', str(test_dir), '-v', '--tb=short'],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        duration = time.time() - start_time
        passed = result.returncode == 0
        
        # Extract test count from output
        error_msg = ""
        if not passed:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'FAILED' in line or 'ERROR' in line:
                    error_msg = line
                    break
        
        return passed, duration, error_msg
        
    except Exception as e:
        duration = time.time() - start_time
        return False, duration, str(e)


def validate_notebooks(root_dir: Path) -> List[Tuple[str, bool, float, str]]:
    """
    Validate all Jupyter notebooks by executing them.
    
    Args:
        root_dir: Root directory to search for notebooks
        
    Returns:
        List of (notebook_name, passed, duration, error_message)
    """
    results = []
    
    # Find all notebooks
    notebooks = list(root_dir.glob('**/*.ipynb'))
    
    # Filter out checkpoint files
    notebooks = [nb for nb in notebooks if '.ipynb_checkpoints' not in str(nb)]
    
    if not notebooks:
        return [("No notebooks found", True, 0.0, "")]
    
    for notebook in sorted(notebooks):
        rel_path = notebook.relative_to(root_dir)
        
        start_time = time.time()
        
        try:
            # Use nbconvert to execute notebook
            result = subprocess.run(
                [
                    'jupyter', 'nbconvert',
                    '--to', 'notebook',
                    '--execute',
                    '--ExecutePreprocessor.timeout=180',
                    '--output', '/tmp/temp.ipynb',
                    str(notebook)
                ],
                capture_output=True,
                text=True,
                timeout=200
            )
            
            duration = time.time() - start_time
            passed = result.returncode == 0
            
            error_msg = ""
            if not passed:
                # Try to extract error from stderr
                if result.stderr:
                    lines = result.stderr.split('\n')
                    for line in lines:
                        if 'Error' in line or 'Exception' in line:
                            error_msg = line[:100]  # Truncate long errors
                            break
            
            results.append((str(rel_path), passed, duration, error_msg))
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            results.append((str(rel_path), False, duration, "Timeout"))
        except Exception as e:
            duration = time.time() - start_time
            results.append((str(rel_path), False, duration, str(e)[:100]))
    
    return results


def validate_imports() -> Tuple[bool, float, str]:
    """
    Test that all critical imports work.
    
    Returns:
        (passed, duration, error_message)
    """
    start_time = time.time()
    
    critical_imports = [
        'torch',
        'numpy',
        'matplotlib',
        'tqdm',
        'pytest',
        'pandas',
    ]
    
    failed_imports = []
    
    for module_name in critical_imports:
        try:
            __import__(module_name)
        except ImportError:
            failed_imports.append(module_name)
    
    duration = time.time() - start_time
    
    if failed_imports:
        error_msg = f"Missing: {', '.join(failed_imports)}"
        return False, duration, error_msg
    else:
        return True, duration, ""


def main():
    """Run full validation suite."""
    parser = argparse.ArgumentParser(description='Validate Tiny Transformer Course')
    parser.add_argument('--tests-only', action='store_true', help='Only run tests')
    parser.add_argument('--notebooks-only', action='store_true', help='Only validate notebooks')
    parser.add_argument('--skip-notebooks', action='store_true', help='Skip notebook validation')
    
    args = parser.parse_args()
    
    # Get repository root
    repo_root = Path(__file__).parent.parent
    
    print_header("Tiny Transformer Course - Validation Suite")
    
    all_passed = True
    
    # 1. Check imports
    if not args.notebooks_only:
        print(f"\n{Colors.BOLD}1. Checking Python Imports{Colors.END}")
        passed, duration, error = validate_imports()
        print_result("Critical imports", passed, duration, error)
        all_passed = all_passed and passed
    
    # 2. Run all tests
    if not args.notebooks_only:
        print(f"\n{Colors.BOLD}2. Running Unit Tests{Colors.END}")
        
        # Find all modules with tests
        modules = [
            d for d in repo_root.iterdir()
            if d.is_dir() and d.name.startswith(('01_', '02_', '03_', '04_', '05_',
                                                  '06_', '07_', '08_', '09_'))
        ]
        
        for module_dir in sorted(modules):
            passed, duration, error = run_module_tests(module_dir)
            print_result(f"{module_dir.name}", passed, duration, error)
            all_passed = all_passed and passed
    
    # 3. Validate notebooks
    if not args.tests_only and not args.skip_notebooks:
        print(f"\n{Colors.BOLD}3. Validating Notebooks{Colors.END}")
        print(f"{Colors.YELLOW}  (This may take a few minutes...){Colors.END}")
        
        notebook_results = validate_notebooks(repo_root)
        
        for nb_name, passed, duration, error in notebook_results:
            print_result(nb_name, passed, duration, error)
            all_passed = all_passed and passed
    
    # Final summary
    print_header("Validation Summary")
    
    if all_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ All validations passed!{Colors.END}\n")
        print("Your environment is ready and all code is working correctly.")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ Some validations failed.{Colors.END}\n")
        print("Please review the errors above and fix them before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
