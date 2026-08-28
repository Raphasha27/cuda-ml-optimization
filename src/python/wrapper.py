"""Python wrapper for CUDA operations"""
import numpy as np
from typing import Optional


class CUDAOptimizer:
    def __init__(self):
        self.available = self._check_cuda()
    
    def _check_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def matrix_multiply_cpu(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return np.dot(A, B)
    
    def matrix_multiply_gpu(self, A: np.ndarray, B: np.ndarray) -> Optional[np.ndarray]:
        if not self.available:
            raise RuntimeError("CUDA not available")
        
        try:
            import torch
            A_gpu = torch.from_numpy(A).cuda()
            B_gpu = torch.from_numpy(B).cuda()
            C_gpu = torch.mm(A_gpu, B_gpu)
            return C_gpu.cpu().numpy()
        except Exception as e:
            raise RuntimeError(f"GPU computation failed: {e}")
    
    def benchmark(self, N: int = 1024) -> dict:
        A = np.random.randn(N, N).astype(np.float32)
        B = np.random.randn(N, N).astype(np.float32)
        
        import time
        
        start = time.time()
        C_cpu = self.matrix_multiply_cpu(A, B)
        cpu_time = time.time() - start
        
        result = {"cpu_time": cpu_time, "cpu_result": C_cpu}
        
        if self.available:
            start = time.time()
            C_gpu = self.matrix_multiply_gpu(A, B)
            gpu_time = time.time() - start
            result["gpu_time"] = gpu_time
            result["speedup"] = cpu_time / gpu_time if gpu_time > 0 else 0
        
        return result
