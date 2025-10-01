#!/usr/bin/env python3

import torch
import numpy as np

def test_torch_matrix_multiply():
    """Test PyTorch matrix multiplication with CPU and GPU (if available)"""
    
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA devices:", torch.cuda.device_count())
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(0))
    
    # Create test matrices
    print("\n--- CPU Matrix Multiplication ---")
    a = torch.randn(3, 4)
    b = torch.randn(4, 5)
    
    print("Matrix A shape:", a.shape)
    print("Matrix B shape:", b.shape)
    
    # CPU matrix multiplication
    c_cpu = torch.matmul(a, b)
    print("Result C shape:", c_cpu.shape)
    print("Result C (CPU):")
    print(c_cpu)
    
    # Test GPU if available
    if torch.cuda.is_available():
        try:
            print("\n--- GPU Matrix Multiplication ---")
            a_gpu = a.cuda()
            b_gpu = b.cuda()
            
            print("Moved matrices to GPU")
            c_gpu = torch.matmul(a_gpu, b_gpu)
            print("GPU computation successful")
            print("Result C (GPU):")
            print(c_gpu)
            
            # Verify results match
            c_gpu_cpu = c_gpu.cpu()
            if torch.allclose(c_cpu, c_gpu_cpu, rtol=1e-5):
                print("✓ CPU and GPU results match!")
            else:
                print("✗ CPU and GPU results differ")
                
        except Exception as e:
            print(f"GPU computation failed: {e}")
    else:
        print("\n--- GPU Not Available ---")
        print("Skipping GPU test")

if __name__ == "__main__":
    try:
        test_torch_matrix_multiply()
        print("\n✓ PyTorch test completed successfully")
    except Exception as e:
        print(f"\n✗ PyTorch test failed: {e}")
        import traceback
        traceback.print_exc()