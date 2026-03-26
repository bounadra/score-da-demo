#!/usr/bin/env python
"""Simple script to display the first 10 lines of test.h5"""

import h5py
from pathlib import Path

# Path to test.h5
with h5py.File("test.h5", 'r') as f:
    # List available datasets
    print("Available datasets:", list(f.keys()))
    print()
    
    # Try to read 'x' dataset (trajectories)
    if 'x' in f:
        x = f['x']
        print(f"Dataset 'x' shape: {x.shape}")
        print(f"Dataset 'x' dtype: {x.dtype}")
        print()
        
        # Display first 10 lines
        print("First 10 lines of 'x':")
        print("-" * 80)
        for i in range(min(10, x.shape[0])):
            print(f"Line {i}: {x[i][:10]}...")  # Show first 10 elements of each line
    else:
        print("'x' dataset not found")
        print("Available keys:", list(f.keys()))
