#usr JiangYu
import numpy as np
from skimage import filters,measure,morphology
from tqdm import tqdm
from ConBased.ConBased_3D_Funs import *

def Get_LBV(coords):
    x_min = coords[:, 0].min()
    x_max = coords[:, 0].max()
    y_min = coords[:, 1].min()
    y_max = coords[:, 1].max()
    z_min = coords[:, 2].min()
    z_max = coords[:, 2].max()
    v_delta = x_max - x_min + 1
    box_data = np.zeros([y_max - y_min + 3, z_max - z_min + 3])
    box_data[coords[:, 1] - y_min + 1, coords[:, 2] - z_min + 1] = 1
    box_label = measure.label(box_data)
    box_region = measure.regionprops(box_label)
    lb_area = box_region[0].area
    return lb_area, v_delta, box_data

def Get_Regions_LBV(origin_data, RMS, RegionMin=8, threshold='otsu'):
    regions = []
    kopen_radius = 1
    n_dilation = 1
    core_data = np.zeros_like(origin_data)
    if threshold == 'mean':
        threshold = origin_data.mean()
    elif threshold == 'otsu':
        threshold = filters.threshold_otsu(origin_data)
    else:
        threshold = threshold
    open_data = morphology.opening(origin_data > threshold, morphology.ball(kopen_radius))
    if n_dilation == 0:
        dilation_data = open_data
    else:
        dilation_data = morphology.dilation(open_data, morphology.ball(kopen_radius))
        dilation_data = dilation_data * (origin_data > RMS)
    dilation_label = measure.label(dilation_data)
    dilation_regions = measure.regionprops(dilation_label)
    for region in dilation_regions:
        coords = region.coords
        lb_area, v_delta, box_data = Get_LBV(coords)
        if lb_area > RegionMin[0] and v_delta > RegionMin[1]:
            regions.append(region)
    for i in range(len(regions)):
        label_x = regions[i].coords[:, 0]
        label_y = regions[i].coords[:, 1]
        label_z = regions[i].coords[:, 2]
        core_data[label_x, label_y, label_z] = origin_data[label_x, label_y, label_z]
    return regions, core_data

def Del_CP_Dict_LBV(core_data, peak_dict, core_dict, index, ClumpMin, DIntensity, DDistance, RegionMin):
    item_core = -1
    item_peak = -1
    i_num = index
    dist = Dists_Array([peak_dict[i_num]], list(peak_dict.values()))
    index_sort = np.argsort(dist[0])[1:]
    pdk_sort = np.array(list(peak_dict.keys()))[index_sort]
    number = 26
    k = 0
    if len(pdk_sort) > 0:
        for j_num in pdk_sort[:number]:
            connectivity = Connectivity(core_dict, i_num, j_num)
            if connectivity == 1:
                break
            k += 1
        if connectivity > 1:
            logic = False
        else:
            order = j_num
            peak_delta = core_data[peak_dict[order][0], peak_dict[order][1], peak_dict[order][2]] \
                         - core_data[peak_dict[i_num][0], peak_dict[i_num][1], peak_dict[i_num][2]]
            lb_area, v_delta, box_data = Get_LBV(np.array(core_dict[i_num]))
            DDistance_lb = np.sqrt((peak_dict[i_num][1] - peak_dict[pdk_sort[k]][1]) ** 2 + (
                        peak_dict[i_num][2] - peak_dict[pdk_sort[k]][2]) ** 2)
            DDistance_v = np.abs(peak_dict[i_num][0] - peak_dict[pdk_sort[k]][0])
            logic = (lb_area < ClumpMin[0] and v_delta < ClumpMin[1]) and \
                    peak_delta > DIntensity or \
                    (lb_area <= RegionMin[0] and v_delta <= RegionMin[1]) or \
                    (DDistance_lb < DDistance[0] and DDistance_v < DDistance[1])
        if logic:
            core_dict[order] = core_dict[order] + core_dict[i_num]
            del core_dict[i_num]
            del peak_dict[i_num]
        else:
            lb_area, v_delta, box_data = Get_LBV(np.array(core_dict[i_num]))
            if lb_area > RegionMin[0] and v_delta > RegionMin[1]:
                item_core = core_dict[i_num]
                item_peak = peak_dict[i_num]
            del core_dict[i_num]
            del peak_dict[i_num]
    return core_dict, peak_dict, item_core, item_peak

def Update_CP_Dict_LBV(peak_dict,core_dict,core_data,ClumpMin,DIntensity,DDistance,RegionMin):
    key = 0
    core_dict_record = {}
    peak_dict_record = {}
    while len(core_dict.keys()) != 1:
        if len(core_dict.keys()) > 1:
            indexs = Get_Index(core_dict)
            i = 0
            for index_i in indexs[0]:
                index = np.array(list(core_dict.keys()))[index_i-i]
                i += 1
                core_dict, peak_dict, item_core, item_peak = \
                    Del_CP_Dict_LBV(core_data,peak_dict,core_dict,index,ClumpMin,DIntensity,DDistance,RegionMin)
                if item_core != -1:
                    core_dict_record[key] = item_core
                    peak_dict_record[key] = item_peak
                key += 1
    key += 1
    core_dict_record[key] = core_dict[list(core_dict.keys())[0]]
    peak_dict_record[key] = peak_dict[list(peak_dict.keys())[0]]
    return core_dict_record,peak_dict_record

def Detect_ConBased_LBV(RMS, Threshold, RegionMin, ClumpMin, DIntensity, DDistance, origin_data):
    key = 0
    core_dict_item = {}
    peak_dict_item = {}
    regions, core_data = Get_Regions_LBV(origin_data, RMS, RegionMin, Threshold)
    for i in tqdm(range(len(regions))):
        region = regions[i]
        core_dict, peak_dict = Build_CP_Dict(core_data, region)
        core_dict, peak_dict_center = Update_CP_Dict_LBV(peak_dict, core_dict, core_data, ClumpMin, DIntensity, DDistance,
                                                     RegionMin)
        for k in core_dict.keys():
            core_dict_item[key] = core_dict[k]
            peak_dict_item[key] = peak_dict_center[k]
            key += 1
    detect_infor_dict = DID_ConBased(peak_dict_item, core_dict_item, origin_data)
    return detect_infor_dict
