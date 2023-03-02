#usr JiangYu
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from ConBased.Evaluate_2D_Funs import *

#The Evaluation function
def Evaluate_3D(origin_center,detect_center,origin_flux,detect_flux,allow_dist):
    delta_flux = []
    matched_flux = []
    delta_dist = []
#     dist_record = [[],[]]
    A = np.stack(origin_center,axis = 0)
    ddf1 = pd.DataFrame({'Cen1': A[:, 0], 'Cen2': A[:, 1], 'Cen3': A[:, 2]})
    B = np.stack(detect_center,axis = 0)
    ddf2 = pd.DataFrame({'Cen1': B[:, 0], 'Cen2': B[:, 1], 'Cen3': B[:, 2]})
    _mdf1, _nomdf1, _mdf2, _nomdf2 = get_MatchFrame(ddf1, ddf2, usecols=['Cen1','Cen2','Cen3'], mindt=allow_dist, dbias=1)
    match_id_i = list(_mdf1.index)
    match_id_j = list(_mdf2.index)
    for i,j in zip(match_id_i,match_id_j):
        dist = ((origin_center[i][0]-detect_center[j][0])**2 + (origin_center[i][1]-detect_center[j][1])**2 +
                (origin_center[i][2]-detect_center[j][2])**2)**(1/2)
#         dist_record[0].append(((origin_center[i][1]-detect_center[j][1])**2 +
        #         (origin_center[i][2]-detect_center[j][2])**2)**(1/2))
#         dist_record[1].append(origin_center[i][0]-detect_center[j][0])
        delta_dist.append(dist)
        delta_flux.append((detect_flux[j]-origin_flux[i])/origin_flux[i])
        matched_flux.append(origin_flux[i])
    TP = len(match_id_i)
    FP = len(detect_center) - TP
    FN = len(origin_center) - TP
    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    F1 = 2*TP/(2*TP + FP + FN)
    return match_id_i,match_id_j,recall,precision,F1,delta_dist,matched_flux,delta_flux

def IOU_3D(origin_center,detect_center,origin_regions,regions_data, allow_dist):
    intersection_record = []
    union_record = []
    A = np.stack(origin_center,axis = 0)
    ddf1 = pd.DataFrame({'Cen1': A[:, 0], 'Cen2': A[:, 1], 'Cen3': A[:, 2]})
    B = np.stack(detect_center,axis = 0)
    ddf2 = pd.DataFrame({'Cen1': B[:, 0], 'Cen2': B[:, 1], 'Cen3': B[:, 2]})
    _mdf1, _nomdf1, _mdf2, _nomdf2 = get_MatchFrame(ddf1, ddf2, usecols=['Cen1','Cen2','Cen3'], mindt=allow_dist, dbias=1)
    match_id_i = list(_mdf1.index)
    match_id_j = list(_mdf2.index)
    for i,j in zip(match_id_i,match_id_j):
        detect_regions = np.where(regions_data==j+1)
        origin_T = np.c_[origin_regions[i][0],origin_regions[i][1],origin_regions[i][2]]
        detect_T = np.c_[detect_regions[0],detect_regions[1],detect_regions[2]]
        dist = Dists_Array(origin_T, detect_T)
        intersection_coords = np.where(dist==0)[0].shape[0]
        total_coords = len(origin_regions[i][0])+len(detect_regions[0])
        intersection_record.append(intersection_coords)
        union_record.append(total_coords-intersection_coords)
#     IOU = np.mean(np.array(intersection_record)/np.array(union_record))
    IOU = intersection_record/np.array(union_record)
    return IOU



