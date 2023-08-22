from random import shuffle
from shutil import copy
import numpy as np
import os
import tormentor
from tormentor.random import Uniform
import ocrodeg
from PIL import Image
from scipy import ndimage
from tqdm import tqdm
from random import shuffle
from random import randrange

tormentor.Wrap.intensity = Uniform(value_range=(0., 1.5))

aug_num = 0
qty_per_im = 2

def warp(image):
    sigma = 10
    noise = ocrodeg.bounded_gaussian_noise(image.shape, sigma, 4.0)
    image = ocrodeg.distort_with_noise(image, noise)
    return image

def augment(im):
    base = np.array(im).astype(float)

    todo = []
    for do_warp in (True, False):
        for structural in (0, 1, 2):
            for gaussian in (0, 1, 2):
                if not do_warp and structural==0 and gaussian==0:
                    continue
                todo.append((do_warp, structural, gaussian))
    shuffle(todo)
    todo = todo[:qty_per_im]
    res = []
    for do_warp, structural, gaussian in todo:
        image = 0+base
        if do_warp: image=warp(image)
        if structural==1: image=ndimage.grey_dilation(image, size=(2, 2), structure=np.ones((2, 2)))
        if structural==2: image=ndimage.grey_erosion(image, size=(2, 2), structure=np.ones((2, 2)))
        if gaussian==1: image=ndimage.gaussian_filter(image, 1+randrange(25)/100)
        if gaussian==2: image=2*image - ndimage.gaussian_filter(image, 1+randrange(25)/100)
        image[image<0]=0
        image[image>255]=255
        res.append(Image.fromarray(image.astype(np.uint8)))
    return res

def find_files(directory):
    lst = os.listdir(directory)
    res = []
    for f in lst:
        ff = os.path.join(directory, f)
        if ff.lower().endswith('.jpg') and os.path.isfile(ff):
            res.append((directory, ff.replace('.jpg', '')))
        elif os.path.isdir(ff):
            res += find_files(ff)
    return res
        
files = find_files('data-aug/train')
existing = set([x.split('/')[-1] for _, x in files])
print(existing)
for directory, filename in tqdm(files):
    if not os.path.exists('%s.txt' % filename):
        continue
    im  = Image.open('%s.jpg' % filename).convert('L')
    ims = augment(im)
    for im in ims:
        while ('%05d' % aug_num) in existing:
            aug_num += 1
        p = '%05d' % aug_num
        existing.add(p)
        im.save(os.path.join(directory, '%s.jpg' % p))
        copy('%s.txt' % filename, os.path.join(directory, '%s.txt' % p))
        copy('%s.cf' % filename, os.path.join(directory, '%s.cf' % p))
        copy('%s.pf' % filename, os.path.join(directory, '%s.pf' % p))


# ~ for im_n in os.listdir('im'):
    # ~ im_p = os.path.join('im', im_n)
    # ~ im = Image.open(im_p).convert('L')
    
    # ~ for i, im in enumerate(augment(im)):
        # ~ im.save(os.path.join('out', '%ds.jpg' % i))
    
    # ~ for t in range(10):
        # ~ sigma = 10
        # ~ noise = ocrodeg.bounded_gaussian_noise(image.shape, sigma, 4.0)
        # ~ image = ocrodeg.distort_with_noise(image, noise)
        
        # ~ imaged = ndimage.grey_dilation(image, size=(2, 2), structure=np.ones((2, 2)))
        # ~ imagee = ndimage.grey_erosion(image, size=(2, 2), structure=np.ones((2, 2)))
        # ~ imageb = ndimage.gaussian_filter(image, 1.25)
        # ~ images = 2*image - ndimage.gaussian_filter(image, 1.25)
        
        # ~ imr = Image.fromarray(image)
        # ~ imr.save(os.path.join('out', '%d.jpg' % t))
        
        # ~ imrd = Image.fromarray(imaged)
        # ~ imrd.save(os.path.join('out', '%dd.jpg' % t))
        
        # ~ imre = Image.fromarray(imagee)
        # ~ imre.save(os.path.join('out', '%de.jpg' % t))
        
        # ~ imrb = Image.fromarray(imageb)
        # ~ imrb.save(os.path.join('out', '%db.jpg' % t))
        
        # ~ imrs = Image.fromarray(images)
        # ~ imrs.save(os.path.join('out', '%ds.jpg' % t))
    # ~ quit()
