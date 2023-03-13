from PIL import Image
import numpy as np
import pdb

img_np_path = 'synthetic.npy'
img_np = np.load(img_np_path) #(480, 640), min: 1.2863141, max:inf
img_np = np.clip(img_np, a_min=0.01, a_max=20)
img_png_path = 'depth/synthetic.png'
pdb.set_trace()
img_png = Image.fromarray(img_np)
img_png.save(img_png_path)