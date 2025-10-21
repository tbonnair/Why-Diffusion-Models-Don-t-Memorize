#!/usr/bin/env python3
"""
Test script to verify the environment is set up correctly.
Run this after creating the conda environment to check all dependencies.
"""

import sys

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...\n")
    
    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'scipy': 'SciPy',
        'PIL': 'Pillow',
        'tqdm': 'tqdm',
        'natsort': 'natsort',
        'pytorch_fid': 'pytorch_fid',
    }
    
    failed = []
    for package, name in packages.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {name:15s} version: {version}")
        except ImportError as e:
            print(f"✗ {name:15s} FAILED to import")
            failed.append(name)
    
    print()
    
    if failed:
        print(f"❌ Failed to import: {', '.join(failed)}")
        return False
    else:
        print("✓ All packages imported successfully!")
        return True

def test_torch():
    """Test PyTorch installation and CUDA availability."""
    print("\nTesting PyTorch...\n")
    
    try:
        import torch
        
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
        else:
            print("Running on CPU (no CUDA GPU detected)")
        
        # Test basic tensor operation
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = x @ y
        print(f"\n✓ Basic tensor operations work!")
        
        if torch.cuda.is_available():
            x_gpu = x.cuda()
            y_gpu = y.cuda()
            z_gpu = x_gpu @ y_gpu
            print(f"✓ GPU tensor operations work!")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False

def test_matplotlib():
    """Test matplotlib plotting."""
    print("\nTesting Matplotlib...\n")
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, ax = plt.subplots()
        x = np.linspace(0, 2*np.pi, 100)
        ax.plot(x, np.sin(x))
        plt.close(fig)
        
        print("✓ Matplotlib plotting works!")
        return True
        
    except Exception as e:
        print(f"❌ Matplotlib test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Environment Verification Test")
    print("=" * 60)
    print()
    
    results = []
    
    # Test imports
    results.append(("Package Imports", test_imports()))
    
    # Test PyTorch
    results.append(("PyTorch", test_torch()))
    
    # Test Matplotlib
    results.append(("Matplotlib", test_matplotlib()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} {name}")
    
    print()
    
    if all(result[1] for result in results):
        print("✓ All tests passed! Environment is ready to use.")
        print()
        print("You can now run the diffusion model code:")
        print("  cd Experiments/src/Training")
        print("  python run_GMM.py -n 4096 -d 8 -s 1 -de 128 -O Adam -B 512 -t -1")
        return 0
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("You may need to reinstall the environment or install missing packages.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
