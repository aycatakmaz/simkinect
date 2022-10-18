import numpy as np 
import cv2 

from scipy.interpolate import griddata
from scipy import ndimage
from camera_utils import vertex_from_depth, normal_from_vertex, add_gaussian_shifts, filterDisp
import pdb

import itertools



if __name__ == "__main__":

    dot_pattern_ = cv2.imread("./data/kinect-pattern_3x3.png", 0)
    focal_length = 554.0
    invalid_disp_ = 99999999.9
    a_max = 20
    #scale_factor = 50
    #baseline_m = 0.3 #75# 0.075 #1
    #std=0.3#1/2.0
    #size_filt_=6#9

    scale_factor_list = [100] #[50,100,200] #list( range(20, 100, 10))+list(range(100, 250, 50)) #11
    baseline_m_list = [0.075, 0.2, 0.3, 0.4] #list(np.arange(0.025, 0.1, 0.025)) #+ [0.2] #, 0.5, 1.0] #list(np.arange(0.2, 1.2, 0.2)) #
    std_list =  [0.1, 0.2, 0.3, 0.5, 0.75, 1]#, 2]#, 5]#list(np.arange(0.1, 1, 0.1)) + list(range(1, 6))
    size_filter = [6,9] #list(range(5, 11))
    full_param_list = [scale_factor_list, baseline_m_list, std_list, size_filter]
    param_combs = list(itertools.product(*full_param_list))
    print(len(param_combs))
    for param_comb in param_combs:
        scale_factor, baseline_m, std, size_filt_ = param_comb
        print(param_comb)
        depth_np_path = 'synthetic.npy'
        depth_orig = np.load(depth_np_path) #(480, 640), min: 1.2863141, max:inf
        depth_orig = np.clip(depth_orig, a_min=0.01, a_max=a_max) #20

        h, w = depth_orig.shape 

        depth_interp = add_gaussian_shifts(depth_orig, std=std)

        disp_= focal_length * baseline_m / (depth_interp + 1e-10)

        depth_f = np.round(disp_ * 8.0)/8.0

        out_disp = filterDisp(depth_f, dot_pattern_, invalid_disp_, size_filt_=size_filt_)

        depth = focal_length * baseline_m / out_disp
        depth[out_disp == invalid_disp_] = 0 
        

        # The depth here needs to converted to cms so scale factor is introduced 
        # though often this can be tuned from [100, 200] to get the desired banding / quantisation effects 


        noisy_depth = (35130/np.round((35130/(np.round(depth*scale_factor) + 1e-10)) + np.random.normal(size=(h, w))*(1.0/6.0) + 0.5))/scale_factor 

        cv2.imwrite('written.png', np.hstack((depth, noisy_depth)))
        #cv2.imwrite('_'.join(['written_high_contrast',str(scale_factor), str(baseline_m), str(std), str(size_filt_)])+'.png', np.hstack((depth_orig*10, noisy_depth*10)))
        cv2.imwrite('_'.join(['whc', str(scale_factor).zfill(3), '{0:.3f}'.format(baseline_m), '{0:.2f}'.format(std), str(size_filt_).zfill(2)])+'.png', np.hstack((depth_orig*30, noisy_depth*30)))

