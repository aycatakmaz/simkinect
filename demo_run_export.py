import numpy as np 
import cv2 

from scipy.interpolate import griddata
from scipy import ndimage
from camera_utils import vertex_from_depth, normal_from_vertex, add_gaussian_shifts, filterDisp
import pdb

import itertools

def add_kinect_noise_to_depth(depth_orig, 
                            dot_pattern,
                            scale_factor=100, 
                            baseline_m=0.075, 
                            std=0.5,
                            size_filt=6,
                            focal_length = 554.0,
                            invalid_disp = 99999999.9,
                            a_min = 0.01, #near_plane
                            a_max = 20, #far_plane
                            w = 640,
                            h = 480):

        #depth_orig = np.load(depth_np_path) #(480, 640), min: 1.2863141, max:inf
        depth_invalid_exch = depth_orig.copy()
        depth_invalid_exch[depth_invalid_exch==float('inf')] = invalid_disp
        depth_clipped = depth_invalid_exch
        depth_clipped = np.clip(depth_orig, a_min=a_min, a_max=a_max)
        depth_interp = add_gaussian_shifts(depth_clipped, std=std)
        disp_= focal_length * baseline_m / (depth_interp + 1e-10)
        depth_f = np.round(disp_ * 8.0)/8.0
        out_disp = filterDisp(depth_f, dot_pattern, invalid_disp, size_filt_=size_filt)
        depth = focal_length * baseline_m / (out_disp + 1e-10)
        depth[out_disp == invalid_disp] = 0
        #depth[out_disp == invalid_disp] = float('inf') 
        
        # The depth here needs to be converted to cms so scale factor is introduced 
        # though often this can be tuned from [100, 200] to get the desired banding / quantisation effects 
        noisy_depth = (35130/np.round((35130/(np.round(depth*scale_factor) + 1e-10)) + np.random.normal(size=(h, w))*(1.0/6.0) + 0.5))/scale_factor 
        
        #pdb.set_trace()
        noisy_depth[noisy_depth<a_min] = float('inf')
        noisy_depth[noisy_depth>=17] = float('inf')
        #noisy_depth[noisy_depth==float('-inf')] = float('inf')
        #noisy_depth[noisy_depth==float('inf')] = 20
        depth_orig[depth_orig==float('inf')] = 0

        depth_factor = 60
        cv2.imwrite('figure_depth.png', depth_orig*depth_factor)
        cv2.imwrite('figure_depth_w_kinect_noise.png', noisy_depth*depth_factor)
        cv2.imwrite('_'.join(['figure_high_contrast',str(scale_factor), str(baseline_m), str(std), str(size_filt)])+'.png', np.hstack((depth_orig*depth_factor, noisy_depth*depth_factor)))

        im_gray = cv2.imread('figure_depth_w_kinect_noise.png', cv2.IMREAD_GRAYSCALE)
        im_c = cv2.applyColorMap(im_gray, cv2.COLORMAP_MAGMA)
        cv2.imwrite('figure_depth_w_kinect_noise_im_c.png', im_c)

        im_gray = cv2.imread('figure_depth.png', cv2.IMREAD_GRAYSCALE)
        im_c = cv2.applyColorMap(im_gray, cv2.COLORMAP_MAGMA)
        cv2.imwrite('figure_depth_im_c.png', im_c)


if __name__ == "__main__":

    dot_pattern = cv2.imread("./data/kinect-pattern_3x3.png", 0)

    depth_np_path = 'figure.npy'
    img_depth_np = np.load(depth_np_path) #(480, 640), min: 1.2863141, max:inf

    img_depth_np_w_kinect_noise = add_kinect_noise_to_depth(img_depth_np, dot_pattern)
    #pdb.set_trace()
    #img_depth_np = img_depth_np_w_kinect_noise 
        
     