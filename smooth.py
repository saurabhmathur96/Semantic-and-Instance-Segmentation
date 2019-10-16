import argparse
from PIL import Image
import numpy as np
from skimage.color import gray2rgb, rgb2gray
from pydensecrf.densecrf import DenseCRF2D
from pydensecrf import densecrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian



def smooth(image, mask):
    # Convert "grayscale" mask to rgb
    mask = gray2rgb(mask)

    # Convert mask to 32 bit RGB
    mask =  mask[:,:,0] + (mask[:,:,1]<<8) + (mask[:,:,2]<<16)

    colors, labels = np.unique(mask, return_inverse=True)

    # Reverse mapping color-map
    colormap = np.zeros([len(colors), 3], np.uint8)
    colormap[:,0] = (colors & 0x0000FF)
    colormap[:,1] = (colors & 0x00FF00) >> 8
    colormap[:,2] = (colors & 0xFF0000) >> 16
    
    n = len(set(labels.flat))
    print ("Total %d labels" % n)

    h, w = image.shape[:2]
    print ("%dx%d image" % (h, w))
    
    crf = DenseCRF2D(w, h, n)

    # -log(P)
    unary = unary_from_labels(mask, n, gt_prob=0.7, zero_unsure=False)
    crf.setUnaryEnergy(unary)

    # Color independent term, features = (x, y)
    crf.addPairwiseGaussian(sxy=(3,3), compat=3, kernel=densecrf.DIAG_KERNEL, normalization=densecrf.NORMALIZE_SYMMETRIC)

    # Color dependent term, features = (x, y, r, g, b)
    crf.addPairwiseBilateral(sxy=(80,80), srgb=(13,13,13), rgbim=image, compat=10, kernel=densecrf.DIAG_KERNEL, normalization=densecrf.NORMALIZE_SYMMETRIC)

    # inference for 5 steps
    Q = crf.inference(5)

    # MaP inference
    map = np.argmax(Q, axis=0)
    
    # Convert back to original colors
    map = colormap[map, :] 
    return map.reshape(image.shape)[:, :, 0]

parser = argparse.ArgumentParser()
parser.add_argument("image", help="path/to/image")
parser.add_argument("mask", help="path/to/mask")
parser.add_argument("output", help="path/to/output")

args = parser.parse_args()
image = np.array(Image.open(args.image), dtype=np.uint8)
mask = np.array(Image.open(args.mask), dtype=np.uint8)

output = smooth(image, mask)
import matplotlib.pyplot as plt


plt.subplot(121)
plt.axis("off")
plt.title("SegNet Output")
plt.imshow(mask)
plt.subplot(122)
plt.axis("off")
plt.title("After DenseCRF post-processing")
plt.imshow(output)
plt.show()
Image.fromarray(output).save(args.output)
