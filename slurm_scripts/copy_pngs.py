import glob
import os
import shutil

for png_path in sorted(glob.glob("lycrt_runs/**/picture.png")):
    snapdir = os.path.dirname(png_path)
    snapname = os.path.basename(snapdir)

    shutil.copyfile(png_path, '/ufrc/narayanan/kimockb/desika_stuff/'+snapname+'.png')

