import os
import glob

cores = sorted(glob.glob('core.*'))
for c in cores:
    print(c)
    os.system('gdb ~/colt-ben/bin/colt.exe {} -x print_rank -batch 2> /dev/null | rg "1 = "'.format(c))
