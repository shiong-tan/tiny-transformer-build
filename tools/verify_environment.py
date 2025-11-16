#!/usr/bin/env python3
"""
Environment Verification Script for Tiny Transformer Course

This script checks that your environment is properly configured:
- Python version
- PyTorch installation and device availability
- Required packages
- Basic tensor operations

Adapted from yhilpisch/llmcode best practices.
"""

import sys
import platform
from typing import List, Tuple


def print_header(text: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print('=' * 70)


def print_result(check: str, passed: bool, details: str = "") -> None:
    """Print a check result with color coding."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} - {check}")
    if details:
        print(f"      {details}")


def check_python_version() -> Tuple[bool, str]:
    """Verify Python version is 3.11+."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major == 3 and version.minor >= 11:
        return True, f"Python {version_str}"
    else:
        return False, f"Python {version_str} (Need 3.11+)"


def check_platform() -> Tuple[bool, str]:
    """Check platform information."""
    system = platform.system()
    machine = platform.machine()
    release = platform.release()
    
    details = f"{system} {release} ({machine})"
    
    # For macOS, provide additional info
    if system == "Darwin":
        mac_ver = platform.mac_ver()[0]
        details = f"macOS {mac_ver} ({machine})"
    
    return True, details


def check_pytorch() -> Tuple[bool, str]:
    """Verify PyTorch installation."""
    try:
        import torch
        version = torch.__version__
        return True, f"PyTorch {version}"
    except ImportError:
        return False, "PyTorch not installed"


def check_device() -> Tuple[bool, str]:
    """Check available compute devices."""
    try:
        import torch
        
        devices = []
        
        # Check CUDA
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            device_name = torch.cuda.get_device_name(0)
            devices.append(f"CUDA {cuda_version} ({device_name})")
        
        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices.append("MPS (Apple Silicon)")
        
        # CPU is always available
        devices.append("CPU")
        
        if len(devices) > 1:
            return True, ", ".join(devices)
        else:
            return True, "CPU only (training will be slower)"
            
    except Exception as e:
        return False, f"Error checking devices: {str(e)}"


def check_required_packages() -> List[Tuple[str, bool, str]]:
    """Check all required packages."""
    required = [
        'numpy',
        'matplotlib',
        'tqdm',
        'pytest',
        'pandas',
        'yaml',
        'jupyter',
    ]
    
    results = []
    for package in required:
        try:
            # Special handling for yaml package name
            import_name = 'yaml' if package == 'yaml' else package
            mod = __import__(import_name)
            
            # Get version if available
            version = getattr(mod, '__version__', 'unknown')
            results.append((package, True, f"v{version}"))
        except ImportError:
            results.append((package, False, "Not installed"))
    
    return results


def test_basic_operations() -> Tuple[bool, str]:
    """Test basic tensor operations."""
    try:
        import torch
        
        # Create tensors
        x = torch.randn(10, 20)
        y = torch.randn(20, 30)
        
        # Matrix multiplication
        z = x @ y
        
        # Check shape
        if z.shape == (10, 30):
            return True, "Matrix operations work correctly"
        else:
            return False, f"Unexpected result shape: {z.shape}"
            
    except Exception as e:
        return False, f"Error: {str(e)}"


def test_autograd() -> Tuple[bool, str]:
    """Test automatic differentiation."""
    try:
        import torch
        
        x = torch.randn(5, 5, requires_grad=True)
        y = x.sum()
        y.backward()
        
        if x.grad is not None and x.grad.shape == x.shape:
            return True, "Autograd works correctly"
        else:
            return False, "Gradient computation failed"
            
    except Exception as e:
        return False, f"Error: {str(e)}"


def main():
    """Run all verification checks."""
    print_header("Tiny Transformer Course - Environment Verification")
    
    print("\n📋 System Information:")
    passed, details = check_platform()
    print_result("Platform", passed, details)
    
    passed, details = check_python_version()
    print_result("Python Version", passed, details)
    
    print("\n🔧 PyTorch Configuration:")
    passed, details = check_pytorch()
    print_result("PyTorch Installation", passed, details)
    
    if passed:
        device_passed, device_details = check_device()
        print_result("Available Devices", device_passed, device_details)
    
    print("\n📦 Required Packages:")
    package_results = check_required_packages()
    all_packages_ok = all(passed for _, passed, _ in package_results)
    
    for package, passed, details in package_results:
        print_result(package, passed, details)
    
    print("\n🧪 Functionality Tests:")
    passed, details = test_basic_operations()
    print_result("Tensor Operations", passed, details)
    
    passed, details = test_autograd()
    print_result("Automatic Differentiation", passed, details)
    
    # Final summary
    print_header("Verification Summary")
    
    if all_packages_ok:
        print("\n✅ Your environment is ready!")
        print("\nNext steps:")
        print("  1. Start with Module 00: cd 00_setup")
        print("  2. Run the setup walkthrough: jupyter notebook setup_walkthrough.ipynb")
        print("  3. Or jump straight to Module 01: cd 01_attention")
    else:
        print("\n⚠️  Some packages are missing.")
        print("\nTo fix:")
        print("  1. Ensure you're in the virtual environment: source .venv/bin/activate")
        print("  2. Install missing packages: pip install -r requirements.txt")
        print("  3. For PyTorch, follow: https://pytorch.org/get-started/locally/")
        print("  4. Run this script again: python setup/verify_environment.py")
    
    print()


if __name__ == "__main__":
    main()
