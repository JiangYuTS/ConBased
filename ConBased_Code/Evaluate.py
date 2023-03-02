#usr JiangYu
import numpy as np
from ConBased.Evaluate_2D_Funs import Evaluate_2D
from ConBased.Evaluate_2D_Funs import IOU_2D
from ConBased.Evaluate_3D_Funs import Evaluate_3D
from ConBased.Evaluate_3D_Funs import IOU_3D

#Evaluate Code
if __name__ == '__main__':
    allow_dist = 2
    infor_dict = 'Generate'
    origin_center = infor_dict['clump_center']
    origin_flux = infor_dict['clump_sum']
    origin_regions = infor_dict['clump_regions']

    detect_center = 'did_ConBased[clump_com]'
    detect_flux = 'did_ConBased[clump_sum]'
    regions_data = 'did_ConBased[regions_data]'


    recall,precision,F1,dist_record,delta_flux,match_flux = \
                    Evaluate_2D(origin_center,detect_center,origin_flux,detect_flux,allow_dist)
    IOU = IOU_2D(origin_center,detect_center,origin_regions,regions_data,allow_dist)
    print('ConBased recall:',recall)
    print('ConBased precision:',precision)
    print('ConBased F1:',F1)
    print('ConBased Dist:',np.mean(dist_record))
    print('ConBased Flux:',np.mean(delta_flux))
    print('ConBased IOU:',np.mean(IOU))

    recall, precision, F1, dist_record, delta_flux, match_flux = \
        Evaluate_3D(origin_center, detect_center, origin_flux, detect_flux, allow_dist)
    IOU = IOU_3D(origin_center,detect_center,origin_regions,regions_data, allow_dist)
    print('ConBased recall:', recall)
    print('ConBased precision:', precision)
    print('ConBased F1:', F1)
    print('ConBased Dist:', np.mean(dist_record))
    print('ConBased Flux:', np.mean(delta_flux))
    print('ConBased IOU:', np.mean(IOU))
