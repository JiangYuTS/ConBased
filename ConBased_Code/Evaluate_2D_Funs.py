#usr JiangYu
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

def get_MatchFrame(ddf1, ddf2, usecols=None, mindt=2.0, dbias=3600):
    """
    从df2中找到df1的匹配
    Args:
        ddf1:
        ddf2:
        usecols:
        mindt:
        dbias:
    Returns:

    """
    df1 = ddf1.copy()
    df2 = ddf2.copy()
    if not isinstance(df1, pd.DataFrame):
        raise RuntimeError('>>> Input df1 must be pandas DataFrame!')
    if not isinstance(df2, pd.DataFrame):
        raise RuntimeError('>>> Input df2 must be pandas DataFrame!')
    if usecols is None:
        usecols = ['RAJ2000', 'DEJ2000']
    else:
        usecols = list(usecols)

    for tmclm in usecols:
        if tmclm not in df1.columns or tmclm not in df2.columns:
            raise RuntimeError('>>> Column {} not in frame!'.format(tmclm))
    # # sort
    # df1.sort_values(by=usecols, inplace=True, ignore_index=True)
    # df2.sort_values(by=usecols, inplace=True, ignore_index=True)

    # create KDTree
    mcoors = df2[usecols].values.copy()
    index2 = np.arange(len(mcoors))
    ocoors = df1[usecols].values.copy()
    index1 = np.arange(len(ocoors))

    myKDtree = cKDTree(mcoors)
    dists, indexs = myKDtree.query(ocoors)
    dists *= dbias
    # create distance matrix
    indexArr = np.vstack((dists, indexs))
    indexArr = np.vstack((indexArr, index1)).T
    # selection step 1, distance cut
    dists = indexArr[:, 0]
    _midx = np.argwhere(dists < mindt)
    if len(_midx) > 0:
        _midx = _midx[:, 0]
    else:
        return [None, None, None, None]
    sdistarr = indexArr[_midx, :]
    sdistarr = pd.DataFrame(data=sdistarr, columns=['dist', 'mids', 'oids'])
    sdistarr1 = sdistarr.groupby('mids').filter(lambda x: len(x) <= 1)
    sdistarr1.reset_index(drop=True, inplace=True)
    sdistarr2 = sdistarr.groupby('mids').filter(lambda x: len(x) > 1)

    # find duplicated matches
    if len(sdistarr2) > 0:
        sdistarr2.sort_values(by=['mids', 'dist'], inplace=True, ignore_index=True)
        sdistarr2.reset_index(drop=True, inplace=True)
        # _dupids = sdistarr2.groupby('mids').groups
        sdistarr3 = sdistarr2.groupby('mids').head(1)
        sdistarr1 = sdistarr1.append(sdistarr3, ignore_index=True)
    else:
        # no duplicated match
        pass

    _midx1 = sdistarr1['oids'].values.astype(np.int_)
    _mdf1 = df1.iloc[_midx1].copy()
    if len(_mdf1) == 0: _mdf1 = None
    _noidx1 = np.setdiff1d(index1, _midx1)
    _nomdf1 = df1.iloc[_noidx1].copy()
    if len(_nomdf1) == 0: _nomdf1 = None

    _midx2 = sdistarr1['mids'].values.astype(np.int_)
    _mdf2 = df2.iloc[_midx2].copy()
    if len(_mdf2) == 0: _mdf2 = None
    _noidx2 = np.setdiff1d(index2, _midx2)
    _nomdf2 = df2.iloc[_noidx2].copy()
    if len(_nomdf2) == 0: _nomdf2 = None
    return [_mdf1, _nomdf1, _mdf2, _nomdf2]

def Dists_Array(matrix_1, matrix_2):
    matrix_1 = np.array(matrix_1)
    matrix_2 = np.array(matrix_2)
    dist_1 = -2 * np.dot(matrix_1, matrix_2.T)
    dist_2 = np.sum(np.square(matrix_1), axis=1, keepdims=True)
    dist_3 = np.sum(np.square(matrix_2), axis=1)
    dists = np.sqrt(dist_1 + dist_2 + dist_3)
    return dists

def Evaluate_2D(origin_center,detect_center,origin_flux,detect_flux,allow_dist):
    delta_flux = []
    matched_flux = []
    delta_dist = []
#     dist_record = [[],[]]
    A = np.stack(origin_center,axis = 0)
    ddf1 = pd.DataFrame({'Cen1': A[:, 0], 'Cen2': A[:, 1]})
    B = np.stack(detect_center,axis = 0)
    ddf2 = pd.DataFrame({'Cen1': B[:, 0], 'Cen2': B[:, 1]})
    _mdf1, _nomdf1, _mdf2, _nomdf2 = get_MatchFrame(ddf1, ddf2, usecols=['Cen1','Cen2'], mindt=allow_dist, dbias=1)
    match_id_i = list(_mdf1.index)
    match_id_j = list(_mdf2.index)
    for i,j in zip(match_id_i,match_id_j):
        dist = ((origin_center[i][0]-detect_center[j][0])**2 + (origin_center[i][1]-detect_center[j][1])**2)**(1/2)
#         dist_record[0].append(((origin_center[i][1]-detect_center[j][1])**2 + (origin_center[i][2]-detect_center[j][2])**2)**(1/2))
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

def IOU_2D(origin_center,detect_center,origin_regions,regions_data,allow_dist):
    intersection_record = []
    union_record = []
    A = np.stack(origin_center,axis = 0)
    ddf1 = pd.DataFrame({'Cen1': A[:, 0], 'Cen2': A[:, 1]})
    B = np.stack(detect_center,axis = 0)
    ddf2 = pd.DataFrame({'Cen1': B[:, 0], 'Cen2': B[:, 1]})
    _mdf1, _nomdf1, _mdf2, _nomdf2 = get_MatchFrame(ddf1, ddf2, usecols=['Cen1','Cen2'], mindt=allow_dist, dbias=1)
    match_id_i = list(_mdf1.index)
    match_id_j = list(_mdf2.index)
    for i,j in zip(match_id_i,match_id_j):
        detect_regions = np.where(regions_data==j+1)
        origin_T = np.c_[origin_regions[i][0],origin_regions[i][1]]
        detect_T = np.c_[detect_regions[0],detect_regions[1]]
        dist = Dists_Array(origin_T, detect_T)
        intersection_coords = np.where(dist==0)[0].shape[0]
        total_coords = len(origin_regions[i][0])+len(detect_regions[0])
        intersection_record.append(intersection_coords)
        union_record.append(total_coords-intersection_coords)
#     IOU = np.mean(np.array(intersection_record)/np.array(union_record))
    IOU = intersection_record/np.array(union_record)
    return IOU




