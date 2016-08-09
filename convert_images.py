from glob import glob
import h5py, os, collections
import numpy as np
from scipy.misc import imread

color_count = {
    'chanel' : 2,
    'apple'  : 7,
    'london_underground' : 3,
    'olympic' : 6,
    'pepsi' : 3,
    'starbucks' : 3,
    'harley' : 3,    
}

f_h5 = "image_data.h5"
h5 = h5py.File(f_h5,'w')

F_PNG = glob("source_images/*.png")

print "Found {} images to convert".format(len(F_PNG))

for f_png in F_PNG:
    print "Converting", f_png
    name = os.path.basename(f_png).replace('.png','')

    assert(name in color_count)
    color_n = color_count[name]

    raw = imread(f_png).astype(float)

    # Convert to 0-1 scale
    img  = raw.copy()
    img /= 255.
        
    # Find most common colors
    pixels = img.reshape((-1,4))
    C = collections.Counter(map(tuple,pixels[::20].tolist()))
    colors,_ = zip(*C.most_common(color_n))
    colors = np.array(colors)

    print f_png,'\n', colors

    pixels /= np.linalg.norm(pixels, axis=1).reshape(-1,1)
    pix_colors = colors / np.linalg.norm(colors, axis=1).reshape(-1,1)

    dist = np.array([pixels.dot(px) for px in pix_colors])
    mask = np.argmax(dist,axis=0)

    dx,dy,_ = img.shape
    mask = mask.reshape((dx,dy))

    g = h5.create_group(name)
    g['raw_image'] = raw
    g['label_colors'] = colors
    g['labels'] = mask

