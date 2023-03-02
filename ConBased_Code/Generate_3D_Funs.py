#usr JiangYu
import numpy as np
from scipy.stats import multivariate_normal
from skimage import transform,filters,measure,morphology
from ConBased.Generate_2D_Funs import *
# 三维

def Get_New_Center(core_data, cores_coordinate):
    xres, yres, zres = core_data.shape
    x_center = cores_coordinate[0] + 1
    y_center = cores_coordinate[1] + 1
    z_center = cores_coordinate[2] + 1
    x_arange = np.arange(max(0, x_center - 1), min(xres, x_center + 2))
    y_arange = np.arange(max(0, y_center - 1), min(yres, y_center + 2))
    z_arange = np.arange(max(0, z_center - 1), min(zres, z_center + 2))
    [x, y, z] = np.meshgrid(x_arange, y_arange, z_arange);
    xyz = np.column_stack([x.flat, y.flat, z.flat])
    gradients = core_data[xyz[:, 0], xyz[:, 1], xyz[:, 2]] \
                - core_data[x_center, y_center, z_center]
    g_step = np.where(gradients == gradients.max())[0][0]
    new_center = list(xyz[g_step] - 1)
    return gradients, new_center

def Build_Peak_Dict(core_data, peak_low, len_peak, len_region):
    k = 1
    peak_dict = {}
    peak_dict[k] = []
    mountain_array = np.zeros_like(core_data)
    temp_core_data = np.zeros(tuple(np.array(core_data.shape) + 2))
    label_data = core_data > peak_low
    label = measure.label(label_data)
    regions = measure.regionprops(label)
    if len(regions) == len_region:
        coordinates = regions[0].coords
        for i in range(1, len(regions)):
            coordinates = np.r_[coordinates, regions[i].coords]
        temp_core_data[1:temp_core_data.shape[0] - 1, 1:temp_core_data.shape[1] - 1,
        1:temp_core_data.shape[2] - 1] = core_data
        for i in range(coordinates.shape[0]):
            temp_coords = []
            if mountain_array[coordinates[i][0], coordinates[i][1], coordinates[i][2]] == 0:
                temp_coords.append(coordinates[i].tolist())
                mountain_array[coordinates[i][0], coordinates[i][1], coordinates[i][2]] = k
                gradients, new_center = Get_New_Center(temp_core_data, coordinates[i])
                if gradients.max() > 0 and mountain_array[new_center[0], new_center[1], new_center[2]] == 0:
                    temp_coords.append(new_center)
                while gradients.max() > 0 and mountain_array[new_center[0], new_center[1], new_center[2]] == 0:
                    mountain_array[new_center[0], new_center[1], new_center[2]] = k
                    gradients, new_center = Get_New_Center(temp_core_data, new_center)
                    if gradients.max() > 0 and mountain_array[new_center[0], new_center[1], new_center[2]] == 0:
                        temp_coords.append(new_center)
                mountain_array[np.stack(temp_coords)[:, 0], np.stack(temp_coords)[:, 1], np.stack(temp_coords)[:, 2]] = \
                    mountain_array[new_center[0], new_center[1], new_center[2]]
                if gradients.max() <= 0:
                    peak_dict[k] = new_center
                    k += 1
                    peak_dict[k] = []
        del (peak_dict[k])
        len_peak = len(peak_dict.keys())
    else:
        len_peak += 1
        len_region += 1
    return len_peak, len_region

def Generate_Sigma(n, sigma_one, sigma_two, sigma_three):
    sigma = [[], [], []]
    for i in range(n):
        sigma[0].append(np.random.random(1) * (sigma_one[1] - sigma_one[0]) + sigma_one[0])
        sigma[1].append(np.random.random(1) * (sigma_two[1] - sigma_two[0]) + sigma_two[0])
        sigma[2].append(np.random.random(1) * (sigma_three[1] - sigma_three[0]) + sigma_three[0])
    return sigma

def Generate_Center(n, xres, yres, zres, sigma, ctime):
    x_center = []
    y_center = []
    z_center = []
    length = 0
    while length <= n:
        x_center.append(list(np.random.random(1) * (xres - 2 * ctime * sigma[0]) + ctime * sigma[0])[0])
        y_center.append(list(np.random.random(1) * (yres - 2 * ctime * sigma[1]) + ctime * sigma[1])[0])
        z_center.append(list(np.random.random(1) * (zres - 2 * ctime * sigma[2]) + ctime * sigma[2])[0])
        length = len(x_center)
    return x_center, y_center, z_center

def Update_Prob_Density(prob_density,coords,peak):
    #根据coords更新概率密度prob_density
    pd_data = np.zeros_like(prob_density)
    prob_density = prob_density/prob_density.max()
    pd_data[coords] = prob_density[coords]
    pd_data = pd_data*peak
    return pd_data

def Rotate(angle, z_center, y_center, coords, data):
    for i in range(np.array(coords)[0].min(), np.array(coords)[0].max()):
        rotate_data = transform.rotate(data[i, :, :], angle, center=[z_center, y_center])
        data[i, :, :] = rotate_data
    label_data = data > 0
    label = measure.label(label_data)
    regions = measure.regionprops(label)
    region_t = regions[0].coords
    region = (region_t[:, 0], region_t[:, 1], region_t[:, 2])
    return data, region

def Get_Coords(xyz, x_center, y_center, z_center, sigma, ctime):
    cut = ctime * sigma
    logic = (xyz[:, 0] - x_center) ** 2 / (cut[0] ** 2) \
            + (xyz[:, 1] - y_center) ** 2 / (cut[1] ** 2) \
            + (xyz[:, 2] - z_center) ** 2 / (cut[2] ** 2)
    coords = xyz[logic <= 1]
    coords = (coords[:, 0], coords[:, 1], coords[:, 2])
    return coords

def Gauss_Noise(cov, data):
    mean = 0#3*cov
    noise = np.random.normal(mean, cov,(data.shape[0],data.shape[1],data.shape[2]))
#     noise[np.where(data==0)] = 0
    data = data + noise
    return data

def Generate_3D(n,xres,yres,zres,peak_low,peak_high,ctime,sigma_one,sigma_two,sigma_three,rms,angle=None):
    peak = []
    peak_value = []
    peak_location = []
    angle_t = []
    center = []
    center_record = []
    clump_size = []
    clump_sum = []
    clump_volume = []
    regions = []
    new_regions = []
    infor_dict = {}
    x, y, z = np.mgrid[0:xres:1, 0:yres:1, 0:zres:1]
    xyz = np.column_stack([x.flat, y.flat, z.flat])
    origin_data = np.zeros([xres, yres, zres])
    sigma = Generate_Sigma(n * 5, sigma_one, sigma_two, sigma_three)
    temp_sigma = np.array([np.array(sigma_one).mean(), np.array(sigma_two).mean(), np.array(sigma_three).mean()],
                          dtype='uint8')
    x_center, y_center, z_center = Generate_Center(5 * n, xres, yres, zres, temp_sigma, ctime)
    i = 0
    temp_length = 1
    len_peak = 0
    len_region = 1
    while len(peak_value) < n:
        peak.append(list(np.random.random(1) * (peak_high - peak_low) + peak_low)[0])
        if angle or angle == 0:
            angle_t.append(angle)
        else:
            angle_t.append(np.random.randint(0, 360))
        sigma_t = np.c_[sigma[0][i], sigma[1][i], sigma[2][i]][0]
        covariance = np.diag(sigma_t ** 2)
        center.append([x_center[i], y_center[i], z_center[i]])
        prob_density = multivariate_normal.pdf(xyz, mean=center[i], cov=covariance)
        prob_density = prob_density.reshape(origin_data.shape)
        coords = Get_Coords(xyz, x_center[i], y_center[i], z_center[i], sigma_t, ctime)
        pd_data = Update_Prob_Density(prob_density, coords, peak[i])
        rotate_data, region = Rotate(angle_t[i], z_center[i], y_center[i], coords, pd_data)
        origin_data += rotate_data
        len_peak, len_region = Build_Peak_Dict(origin_data, peak_low, len_peak, len_region)
        if len_peak == temp_length:
            peak_value.append(peak[i])
            temp_length = len(peak_value) + 1
            center_record.append(center[i])

            core_x = region[0]
            core_y = region[1]
            core_z = region[2]
            core_x_min = core_x.min()
            core_x_max = core_x.max()
            core_y_min = core_y.min()
            core_y_max = core_y.max()
            core_z_min = core_z.min()
            core_z_max = core_z.max()
            temp_core = np.zeros(
                (core_x_max - core_x_min + 1, core_y_max - core_y_min + 1, core_z_max - core_z_min + 1))
            temp_core[core_x - core_x_min, core_y - core_y_min, core_z - core_z_min] = origin_data[
                core_x, core_y, core_z]
            peak_coord = np.where(temp_core == temp_core.max())
            peak_coord = [(peak_coord[0] + core_x_min)[0], (peak_coord[1] + core_y_min)[0],
                          (peak_coord[2] + core_z_min)[0]]
            peak_location.append(peak_coord)
            clump_size.append(list(2 * np.sqrt(2 * np.log(2)) * sigma_t))
            clump_sum.append(np.array(rotate_data[region]).sum())
            clump_volume.append(region[0].shape[0])
            regions.append(region)
        else:
            origin_data -= rotate_data
        i += 1
    noise_data = Gauss_Noise(rms, origin_data)
    sorted_id = sorted(range(len(center_record)), key=lambda k: center_record[k], reverse=False)
    infor_dict['peak_value'] = np.array(peak_value)[sorted_id].tolist()
    infor_dict['peak_location'] = np.array(peak_location)[sorted_id].tolist()
    infor_dict['angle'] = np.array(angle_t)[sorted_id].tolist()
    infor_dict['clump_center'] = np.array(center_record)[sorted_id].tolist()
    infor_dict['clump_size'] = np.around(np.array(clump_size)[sorted_id], 3).tolist()
    infor_dict['clump_sum'] = np.around(np.array(clump_sum)[sorted_id], 3).tolist()
    infor_dict['clump_volume'] = np.array(clump_volume)[sorted_id].tolist()
    for i in range(len(sorted_id)):
        new_regions.append(regions[sorted_id[i]])
    infor_dict['clump_regions'] = new_regions
    return origin_data, noise_data, infor_dict


