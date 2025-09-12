# Modules -----------------------------------------------------------------------------------------------------------------#
import os
import psutil
import warnings

import numpy as np

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

# Check for torch availability --------------------------------------------------------------------------------------------#
try:
    import torch
    import torch.cuda
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Configuration for resources ---------------------------------------------------------------------------------------------#
@dataclass
class ResourceConfig:
    """Configuration for resource management"""
    max_parallel_trials : int = 1                     # Max number of parallel trials  
    gpu_memory_limit    : float = 0.9                 # 90% of GPU memory
    cpu_memory_limit    : float = 0.8                 # 80% of RAM
    prefer_gpu          : bool = True                 # Prefer GPU if available
    gpu_ids             : Optional[List[int]] = None  # Specific GPU IDs to use
    n_jobs_per_trial    : int = 1                     # Number of CPU threads per trial

# Simple manager of GPU devices and memory --------------------------------------------------------------------------------#
class DeviceManager:
    """Manages GPU devices and memory"""
    def __init__(self, config: ResourceConfig):
        
        self.config           = config
        self._available_gpus  = self._get_available_gpus()
        self._gpu_allocations = {}  # trial_id -> gpu_id
        
    def _get_available_gpus(self) -> List[int]:
        """Get list of available GPU devices"""
        
        # Return empty if torch is not available
        if not TORCH_AVAILABLE: return []
        
        # If GPU id list is provided, filter valid ones
        if self.config.gpu_ids is not None:
            return [i for i in self.config.gpu_ids if i < torch.cuda.device_count()]
        
        return list(range(torch.cuda.device_count()))

    def get_device(self, trial_id: int) -> str:
        """Get appropriate device for a trial"""
        if not self._available_gpus or not self.config.prefer_gpu:
            return "cpu", None
        
        # Round-robin GPU assignment
        gpu_id = self._available_gpus[trial_id % len(self._available_gpus)]
        self._gpu_allocations[trial_id] = gpu_id

        return f"cuda", gpu_id
    
    def cleanup_trial(self, trial_id: int):
        """Cleanup GPU resources after trial completion"""
        if trial_id in self._gpu_allocations:
            if TORCH_AVAILABLE:
                torch.cuda.empty_cache()
            del self._gpu_allocations[trial_id]

# Simple manager of CPU resources and memory ------------------------------------------------------------------------------#
class CPUManager:
    """Manages CPU resources and threading"""
    def __init__(self, config: ResourceConfig):
        self.config      = config
        self.total_cores = psutil.cpu_count(logical=False)
        
    def get_optimal_threads(self, active_trials: int) -> int:
        """Calculate optimal number of threads for a trial"""
        if active_trials == 0:
            return 1
            
        available_cores = max(1, self.total_cores // active_trials)
        return min(available_cores, self.config.n_jobs_per_trial)
    
    def check_memory(self) -> bool:
        """Check if enough memory is available"""
        memory_percent = psutil.virtual_memory().percent / 100
        return memory_percent < self.config.cpu_memory_limit

# Combine CPU and GPU management ------------------------------------------------------------------------------------------#
class ResourceManager:
    """Master resource manager"""
    def __init__(self, config: ResourceConfig):
        self.config         = config
        self.device_manager = DeviceManager(config)
        self.cpu_manager    = CPUManager(config)
        self._active_trials = 0
        
    def acquire_resources(self, trial_id: int) -> Dict[str, Union[str, int]]:
        """Acquire resources for a new trial"""
        if not self.cpu_manager.check_memory():
            raise RuntimeError("Insufficient system memory")
            
        self._active_trials += 1
        
        return {'device': self.device_manager.get_device(trial_id),
                'n_jobs': self.cpu_manager.get_optimal_threads(self._active_trials)}
    
    def release_resources(self, trial_id: int):
        """Release resources after trial completion"""
        self.device_manager.cleanup_trial(trial_id)
        self._active_trials = max(0, self._active_trials - 1)
    
    @property
    def can_start_trial(self) -> bool:
        """Check if new trial can be started"""
        return (self._active_trials < self.config.max_parallel_trials and 
                self.cpu_manager.check_memory())

    def get_suggested_parallel_trials(self) -> int:
        """Get suggested number of parallel trials based on resources"""
        available_memory = 1 - (psutil.virtual_memory().percent / 100)
        gpu_count = len(self.device_manager._available_gpus)
        
        cpu_based    = max(1, self.cpu_manager.total_cores // self.config.n_jobs_per_trial)
        memory_based = max(1, int(available_memory * self.config.max_parallel_trials))
        gpu_based    = max(1, gpu_count) if self.config.prefer_gpu else cpu_based
        
        return min(cpu_based, memory_based, gpu_based, self.config.max_parallel_trials)

#--------------------------------------------------------------------------------------------------------------------------#