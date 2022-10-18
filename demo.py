import numpy as np 
import cv2 

from scipy.interpolate import griddata
from scipy import ndimage
from camera_utils import vertex_from_depth, normal_from_vertex, add_gaussian_shifts, filterDisp
import pdb

if __name__ == "__main__":

    dot_pattern_ = cv2.imread("./data/kinect-pattern_3x3.png", 0)

    scale_factor = 100 
    focal_length = 554.0
    baseline_m = 1# 0.075 #1


    depth_np_path = 'synthetic.npy'
    depth_orig = np.load(depth_np_path) #(480, 640), min: 1.2863141, max:inf

    depth_orig = np.clip(depth_orig, a_min=0.01, a_max=20)

    h, w = depth_orig.shape 

    depth_interp = add_gaussian_shifts(depth_orig)

    disp_= focal_length * baseline_m / (depth_interp + 1e-10)

    depth_f = np.round(disp_ * 8.0)/8.0

    out_disp = filterDisp(depth_f, dot_pattern_)

    depth = focal_length * baseline_m / out_disp
    depth[out_disp == 99999999.9] = 0 
    

    # The depth here needs to converted to cms so scale factor is introduced 
    # though often this can be tuned from [100, 200] to get the desired banding / quantisation effects 


    noisy_depth = (35130/np.round((35130/np.round(depth*scale_factor)) + np.random.normal(size=(h, w))*(1.0/6.0) + 0.5))/scale_factor 

    cv2.imwrite('written.png', np.hstack((depth, noisy_depth)))
    cv2.imwrite('written_high_contrast_'+str(scale_factor)+'.png', np.hstack((depth_orig*10, noisy_depth*10)))
