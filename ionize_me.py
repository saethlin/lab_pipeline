import numpy as np
import ctypes
import os
import struct

# call the function
exec_call = '/home/kimockb/octreert/octree/src/octree.so'
routine = ctypes.cdll[exec_call]
    
routine.build_octree_from_particle()
 
