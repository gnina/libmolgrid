from .molgrid import *
import sys

#define pytorch specific functionality
try:
    import torch as _torch
    
except ImportError:
    sys.stderr.write("Failed to import torch.\n")