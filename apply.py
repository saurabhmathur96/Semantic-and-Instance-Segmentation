from os import listdir, path, makedirs, symlink
import shutil
import argparse

import numpy as np
from tqdm.auto import tqdm

from psyphy.perturb.ocv import gaussian_blur
from psyphy.perturb.pilie import brightness, color, contrast, sharpness


FUNCTION_MAP = {'gaussian_blur' : gaussian_blur,
                'brightness' : brightness,
                'color': color,
                'contrast': contrast,
                'sharpness': sharpness }


parser = argparse.ArgumentParser()
parser.add_argument('src')
parser.add_argument('dest')
parser.add_argument('f', choices=FUNCTION_MAP.keys())
parser.add_argument('count', type=int)
parser.add_argument('start', type=float)
parser.add_argument('end', type=float)
args = parser.parse_args()

f = FUNCTION_MAP[args.f]
start, end, count = args.start, args.end, args.count
src, dest = args.src, args.dest

makedirs(dest, exist_ok=True)
params = np.logspace(np.log10(start), np.log10(end), count)
params = np.around(params, 4)


for param in params:
    param_path = path.join(dest, '%.4f' % param)
    makedirs(param_path, exist_ok=True)
    annot_path = path.join(dest, '%.4fannot' % param)
    shutil.copytree(src+'annot/', annot_path)
    shutil.copy(src+'.txt', param_path + '.txt')
    for filename in tqdm(listdir(src)):
        rpath = f(path.join(src, filename), param)
        shutil.move(rpath, path.join(param_path, filename))
