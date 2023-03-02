#usr JiangYu
import numpy as np
from skimage import filters,measure,morphology
from tqdm import tqdm

def Get_Regions(origin_data,RMS,region_pixels_min=8,threshold='otsu'):
    regions = []
    kopen_radius=1
    core_data = np.zeros_like(origin_data)
    if threshold == 'mean':
        threshold = origin_data.mean()
    elif threshold == 'otsu':
        threshold = filters.threshold_otsu(origin_data)
    else:
        threshold = threshold
    open_data = morphology.opening(origin_data > threshold,morphology.ball(kopen_radius))
    dilation_data = morphology.dilation(open_data,morphology.ball(kopen_radius))
    dilation_data = dilation_data*(origin_data > threshold)
#     dilation_data = ndimage.binary_fill_holes(dilation_data)
    dilation_label = measure.label(dilation_data)
    dilation_regions = measure.regionprops(dilation_label)
    for region in dilation_regions:
        if region.area > region_pixels_min:
            regions.append(region)
    for i in range(len(regions)):
        label_x = regions[i].coords[:,0]
        label_y = regions[i].coords[:,1]
        label_z = regions[i].coords[:,2]
        core_data[label_x,label_y,label_z] = origin_data[label_x,label_y,label_z]
    return regions,core_data

def Get_New_Center(core_data,cores_coordinate):
    xres,yres,zres = core_data.shape
    x_center = cores_coordinate[0]+1
    y_center = cores_coordinate[1]+1
    z_center = cores_coordinate[2]+1
    x_arange = np.arange(max(0,x_center-1),min(xres,x_center+2))
    y_arange = np.arange(max(0,y_center-1),min(yres,y_center+2))
    z_arange = np.arange(max(0,z_center-1),min(zres,z_center+2))
    [x, y, z] = np.meshgrid(x_arange, y_arange, z_arange);
    xyz = np.column_stack([x.flat, y.flat, z.flat])
    gradients = core_data[xyz[:,0],xyz[:,1],xyz[:,2]]\
                - core_data[x_center,y_center,z_center]
    g_step = np.where(gradients == gradients.max())[0][0]
    new_center = list(xyz[g_step]-1)
    return gradients,new_center

def Build_CP_Dict(core_data,region):
    k = 1
    peak_dict = {}
    peak_dict[k] = []
    mountain_dict = {}
    mountain_dict[k] = []
    core_data = core_data + np.random.random(core_data.shape) / 100000
    mountain_array = np.zeros_like(core_data)
    temp_core_data = np.zeros(tuple(np.array(core_data.shape)+2))
    temp_core_data[1:temp_core_data.shape[0]-1,1:temp_core_data.shape[1]-1,1:temp_core_data.shape[2]-1]=core_data
    coordinates = region.coords
    for i in range(coordinates.shape[0]):
        temp_coords = []
        if mountain_array[coordinates[i][0],coordinates[i][1],coordinates[i][2]] == 0:
            temp_coords.append(coordinates[i].tolist())
            mountain_array[coordinates[i][0],coordinates[i][1],coordinates[i][2]] = k
            gradients,new_center = Get_New_Center(temp_core_data,coordinates[i])
            if gradients.max() > 0 and mountain_array[new_center[0],new_center[1],new_center[2]] == 0:
                temp_coords.append(new_center)
            while gradients.max() > 0 and mountain_array[new_center[0],new_center[1],new_center[2]] == 0:
                mountain_array[new_center[0],new_center[1],new_center[2]] = k
                gradients,new_center = Get_New_Center(temp_core_data,new_center)
                if gradients.max() > 0 and mountain_array[new_center[0],new_center[1],new_center[2]] == 0:
                    temp_coords.append(new_center)
            mountain_array[np.stack(temp_coords)[:,0],np.stack(temp_coords)[:,1],np.stack(temp_coords)[:,2]]=\
                mountain_array[new_center[0],new_center[1],new_center[2]]
            mountain_dict[mountain_array[new_center[0],new_center[1],new_center[2]]] += temp_coords
            if gradients.max() <= 0:
                peak_dict[k] = new_center
                k += 1
                mountain_dict[k] = []
                peak_dict[k] = []
    del(mountain_dict[k])
    del(peak_dict[k])
    core_dict = mountain_dict
    return core_dict,peak_dict

def Connectivity(core_dict,i_num,j_num):
    i_region = np.array(core_dict[i_num])
    j_region = np.array(core_dict[j_num])
    x_min = np.r_[i_region[:,0],j_region[:,0]].min()
    x_max = np.r_[i_region[:,0],j_region[:,0]].max()
    y_min = np.r_[i_region[:,1],j_region[:,1]].min()
    y_max = np.r_[i_region[:,1],j_region[:,1]].max()
    z_min = np.r_[i_region[:,2],j_region[:,2]].min()
    z_max = np.r_[i_region[:,2],j_region[:,2]].max()
    box_data = np.zeros([x_max-x_min+1,y_max-y_min+1,z_max-z_min+1])
    box_data[i_region[:,0]-x_min,i_region[:,1]-y_min,i_region[:,2]-z_min] = 1
    box_data[j_region[:,0]-x_min,j_region[:,1]-y_min,j_region[:,2]-z_min] = 1
    box_label = measure.label(box_data)
    box_region = measure.regionprops(box_label)
    return len(box_region)

def Dists_Array(matrix_1, matrix_2):
    matrix_1 = np.array(matrix_1)
    matrix_2 = np.array(matrix_2)
    dist_1 = -2 * np.dot(matrix_1, matrix_2.T)
    dist_2 = np.sum(np.square(matrix_1), axis=1, keepdims=True)
    dist_3 = np.sum(np.square(matrix_2), axis=1)
    dists = np.sqrt(dist_1 + dist_2 + dist_3)
    return dists

def Get_Index(core_dict):
    mountain_size = []
    for key in core_dict.keys():
        mountain_size.append(len(core_dict[key]))
    mountain_size = np.array(mountain_size)
    indexs = np.where(mountain_size == mountain_size.min())
    return indexs

def Del_CP_Dict(core_data,peak_dict,core_dict,index,ClumpMin,DIntensity,DDistance,RegionMin):
    distance = []
    item_core = -1
    item_peak = -1
    i_num = index
    dist = Dists_Array([peak_dict[i_num]], list(peak_dict.values()))
    index_sort = np.argsort(dist[0])[1:]
    distance_sort = np.array(dist[0])[index_sort]
    pdk_sort = np.array(list(peak_dict.keys()))[index_sort]
    number = 26
    k = 0
    if len(pdk_sort)>0:
        for j_num in pdk_sort[:number]:
            connectivity = Connectivity(core_dict,i_num,j_num)
            if connectivity == 1:
                break
            k += 1
        if connectivity > 1:
            logic = False
        else:
            order = j_num
            nearest_dis_0 = distance_sort[k]
            peak_delta_0 = core_data[peak_dict[order][0],peak_dict[order][1],peak_dict[order][2]]\
                         - core_data[peak_dict[i_num][0],peak_dict[i_num][1],peak_dict[i_num][2]]
            logic = len(core_dict[i_num]) < ClumpMin and peak_delta_0 > DIntensity\
                     or nearest_dis_0 < DDistance or len(core_dict[i_num]) < RegionMin
        if logic:
            core_dict[order] = core_dict[order] + core_dict[i_num]
            del core_dict[i_num]
            del peak_dict[i_num]
        else :
            if len(core_dict[i_num]) > RegionMin:
                item_core = core_dict[i_num]
                item_peak = peak_dict[i_num]
            del core_dict[i_num]
            del peak_dict[i_num]
    return core_dict,peak_dict,item_core,item_peak

def Update_CP_Dict(peak_dict,core_dict,core_data,ClumpMin,DIntensity,DDistance,RegionMin):
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
                core_dict,peak_dict,item_core,item_peak = \
                    Del_CP_Dict(core_data,peak_dict,core_dict,index,ClumpMin,DIntensity,DDistance,RegionMin)
                if item_core != -1:
                    core_dict_record[key] = item_core
                    peak_dict_record[key] = item_peak
                key += 1
    key += 1
    core_dict_record[key] = core_dict[list(core_dict.keys())[0]]
    peak_dict_record[key] = peak_dict[list(peak_dict.keys())[0]]
    return core_dict_record,peak_dict_record

def Get_DV(box_data,box_center):
    #3D
    box_data_sum = box_data.sum(0)
    box_region = np.where(box_data_sum!= 0)
    A11 = np.sum((box_region[0]-box_center[1])**2*\
        box_data_sum[box_region])
    A12 = -np.sum((box_region[0]-box_center[1])*\
        (box_region[1]-box_center[2])*\
        box_data_sum[box_region])
    A21 = A12
    A22 = np.sum((box_region[1]-box_center[2])**2*\
        box_data_sum[box_region])
    A = np.array([[A11,A12],[A21,A22]])/len(box_region[0])
    D, V = np.linalg.eig(A)
    if D[0] < D[1]:
        D = D[[1,0]]
        V = V[[1,0]]
    if V[1][0]<0 and V[0][0]>0 and V[1][1]>0:
        V = -V
    size_ratio = np.sqrt(D[0]/D[1])
    angle = np.around(np.arccos(V[0][0])*180/np.pi-90,2)
    return D,V,size_ratio,angle

def DID_ConBased(center_dict,core_dict,origin_data):
    peak_value = []
    peak_location = []
    clump_maximum=[]
    clump_com = []
    clump_size = []
    clump_sum = []
    clump_volume = []
    clump_regions = []
    clump_edge = []
    clump_angle = []
    single_clump = []
    detect_infor_dict = {}
    k = 0
    regions_data = np.zeros_like(origin_data)
    for key in center_dict.keys():
        k += 1
        core_x = np.array(core_dict[key])[:,0]
        core_y = np.array(core_dict[key])[:,1]
        core_z = np.array(core_dict[key])[:,2]
        core_x_min = core_x.min()
        core_x_max = core_x.max()
        core_y_min = core_y.min()
        core_y_max = core_y.max()
        core_z_min = core_z.min()
        core_z_max = core_z.max()
        temp_core = np.zeros((core_x_max-core_x_min+1,core_y_max-core_y_min+1,core_z_max-core_z_min+1))
        temp_core[core_x-core_x_min,core_y-core_y_min,core_z-core_z_min]=origin_data[core_x,core_y,core_z]
        temp_center = [center_dict[key][0]-core_x_min,center_dict[key][1]-core_y_min,center_dict[key][2]-core_z_min]
        D,V,size_ratio,angle = Get_DV(temp_core,temp_center)
        peak_coord = np.where(temp_core == temp_core.max())
        peak_coord = [(peak_coord[0]+core_x_min)[0],(peak_coord[1]+core_y_min)[0],(peak_coord[2]+core_z_min)[0]]
        peak_value.append(origin_data[peak_coord[0],peak_coord[1],peak_coord[2]])
        peak_location.append(peak_coord)
        clump_maximum.append(center_dict[key])
        od_mass = origin_data[core_x,core_y,core_z]
        mass_array = np.c_[od_mass,od_mass,od_mass]
        clump_com.append(np.around((np.c_[mass_array]*core_dict[key]).sum(0)\
                    /od_mass.sum(),3).tolist())
        size = np.sqrt((mass_array*(np.array(core_dict[key])**2)).sum(0)/od_mass.sum()-\
                       ((mass_array*np.array(core_dict[key])).sum(0)/od_mass.sum())**2)
        clump_size.append(size.tolist())
        clump_sum.append(origin_data[core_x,core_y,core_z].sum())
        clump_volume.append(len(core_dict[key]))
        clump_angle.append(angle)
        regions_data[core_x,core_y,core_z] = k
        # clump_regions.append([np.array(core_dict[key])[:,0],np.array(core_dict[key])[:,1],np.array(core_dict[key])[:,2]])
        data_size = origin_data.shape
        if core_x_min == 0 or core_y_min == 0 or core_z_min == 0 or \
            core_x_max+1 == data_size[0] or core_y_max+1 == data_size[1] or core_z_max+1 == data_size[2]:
            clump_edge.append(1)
        else:
            clump_edge.append(0)
        # single_clump.append(temp_core)
    detect_infor_dict['peak_value'] = list(np.around(peak_value,3))
    detect_infor_dict['peak_location'] = peak_location
    detect_infor_dict['clump_maximum'] = clump_maximum
    detect_infor_dict['clump_center'] = clump_com
    detect_infor_dict['clump_size'] = list(np.around(clump_size,3))
    detect_infor_dict['clump_sum'] = list(np.around(clump_sum,3))
    detect_infor_dict['clump_volume'] = clump_volume
    detect_infor_dict['clump_angle'] = clump_angle
    detect_infor_dict['clump_edge'] = clump_edge
    detect_infor_dict['regions_data'] = regions_data
#     detect_infor_dict['single_clump'] = single_clump
#     detect_infor_dict['clump_regions'] = clump_regions
    return detect_infor_dict

def Detect_ConBased(RMS,Threshold,RegionMin,ClumpMin,DIntensity,DDistance,origin_data):
    key = 0
    core_dict_item = {}
    peak_dict_item = {}
    RegionMin = RegionMin[0]
    ClumpMin = ClumpMin[0]
    DDistance = DDistance[0]
    regions,core_data = Get_Regions(origin_data,RMS,RegionMin,Threshold)
    for i in tqdm(range(len(regions))):
        region = regions[i]
        core_dict,peak_dict = Build_CP_Dict(core_data,region)
        core_dict,peak_dict_center = Update_CP_Dict(peak_dict,core_dict,core_data,ClumpMin,DIntensity,DDistance,RegionMin)
        for k in core_dict.keys():
            core_dict_item[key] = core_dict[k]
            peak_dict_item[key] = peak_dict_center[k]
            key += 1
    detect_infor_dict = DID_ConBased(peak_dict_item,core_dict_item,origin_data)
    return detect_infor_dict