#usr JiangYu
# 2D
import numpy as np
from scipy.stats import multivariate_normal
from skimage import transform,filters,measure,morphology

def Get_New_Center(core_data, cores_coordinate):
    xres, yres = core_data.shape
    x_center = cores_coordinate[0] + 1
    y_center = cores_coordinate[1] + 1
    x_arange = np.arange(max(0, x_center - 1), min(xres, x_center + 2))
    y_arange = np.arange(max(0, y_center - 1), min(yres, y_center + 2))
    [x, y] = np.meshgrid(x_arange, y_arange)
    xy = np.column_stack([x.flat, y.flat])
    gradients = core_data[xy[:, 0], xy[:, 1]] \
                - core_data[x_center, y_center]
    g_step = np.where(gradients == gradients.max())[0][0]
    new_center = list(xy[g_step] - 1)
    return gradients, new_center

def Build_Peak_Dict(core_data, peak_low):
    k = 1
    peak_dict = {}
    peak_dict[k] = []
    mountain_array = np.zeros_like(core_data)
    efficient_data = np.zeros_like(core_data)
    temp_core_data = np.zeros(tuple(np.array(core_data.shape) + 2))
    efficient_coords = np.where(core_data > peak_low)
    efficient_data[efficient_coords] = core_data[efficient_coords]
    coordinates_t = np.where(efficient_data != 0)
    coordinates = np.c_[coordinates_t[0], coordinates_t[1]]
    temp_core_data[1:temp_core_data.shape[0] - 1, 1:temp_core_data.shape[1] - 1] = core_data
    for i in range(coordinates.shape[0]):
        temp_coords = []
        if mountain_array[coordinates[i][0], coordinates[i][1]] == 0:
            temp_coords.append(coordinates[i].tolist())
            mountain_array[coordinates[i][0], coordinates[i][1]] = k
            gradients, new_center = Get_New_Center(temp_core_data, coordinates[i])
            if gradients.max() > 0 and mountain_array[new_center[0], new_center[1]] == 0:
                temp_coords.append(new_center)
            while gradients.max() > 0 and mountain_array[new_center[0], new_center[1]] == 0:
                mountain_array[new_center[0], new_center[1]] = k
                gradients, new_center = Get_New_Center(temp_core_data, new_center)
                if gradients.max() > 0 and mountain_array[new_center[0], new_center[1]] == 0:
                    temp_coords.append(new_center)
            mountain_array[np.stack(temp_coords)[:, 0], np.stack(temp_coords)[:, 1]] = \
                mountain_array[new_center[0], new_center[1]]
            if gradients.max() <= 0:
                peak_dict[k] = new_center
                k += 1
                peak_dict[k] = []
    del (peak_dict[k])
    return peak_dict

def Generate_Sigma(n, sigma_one, sigma_two):
    sigma = [[], []]
    for i in range(n):
        sigma[0].append(np.random.random(1) * (sigma_one[1] - sigma_one[0]) + sigma_one[0])
        sigma[1].append(np.random.random(1) * (sigma_two[1] - sigma_two[0]) + sigma_two[0])
    return sigma

def Generate_Center(n, xres, yres, sigma, ctime):
    x_center = []
    y_center = []
    length = 0
    while length <= n:
        x_center.append(list(np.random.random(1) * (xres - 2 * ctime * sigma[0]) + ctime * sigma[0])[0])
        y_center.append(list(np.random.random(1) * (yres - 2 * ctime * sigma[1]) + ctime * sigma[1])[0])
        length = len(x_center)
    return x_center, y_center

def Get_Coords(xy, x_center, y_center, sigma, ctime):
    cut = ctime * sigma
    logic = (xy[:, 0] - x_center) ** 2 / (cut[0] ** 2) +\
            (xy[:, 1] - y_center) ** 2 / (cut[1] ** 2)
    coords = xy[logic <= 1]
    coords = (coords[:, 0], coords[:, 1])
    return coords

def Update_Prob_Density(prob_density, coords, peak):
    # 根据coords更新概率密度prob_density
    pd_data = np.zeros_like(prob_density)
    prob_density = prob_density / prob_density.max()
    pd_data[coords] = prob_density[coords]
    pd_data = pd_data * peak
    return pd_data

def Gauss_Noise(cov, data):
    mean = 0
    noise = np.random.normal(mean, cov, data.shape)
    data = data + noise
    return data

def Generate_2D(n, xres, yres, peak_low, peak_high, ctime, sigma_one, sigma_two, rms, angle=None):
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
    x, y = np.mgrid[0:xres:1, 0:yres:1]
    xy = np.column_stack([x.flat, y.flat])
    origin_data = np.zeros([xres, yres])
    sigma = Generate_Sigma(10 * n, sigma_one, sigma_two)
    temp_sigma = np.array([np.array(sigma_one).mean(), np.array(sigma_two).mean()], dtype='uint8')
    x_center, y_center = Generate_Center(n * n, xres, yres, temp_sigma, ctime)
    i = 0
    temp_length = 1
    while len(peak_value) < n:
        peak.append(list(np.random.random(1) * (peak_high - peak_low) + peak_low)[0])
        if angle or angle == 0:
            angle_t.append(angle)
        else:
            angle_t.append(np.random.randint(0, 360))
        sigma_t = np.c_[sigma[0][i], sigma[1][i]][0]
        covariance = np.diag(sigma_t ** 2)
        center.append([x_center[i], y_center[i]])
        prob_density = multivariate_normal.pdf(xy, mean=center[i], cov=covariance)
        prob_density = prob_density.reshape(origin_data.shape)

        coords = Get_Coords(xy, x_center[i], y_center[i], sigma_t, ctime)
        pd_data = Update_Prob_Density(prob_density, coords, peak[i])
        rotate_data = transform.rotate(pd_data, angle_t[i], center=[y_center[i], x_center[i]])
        region = np.where(rotate_data != 0)
        origin_data += rotate_data
        peak_dict = Build_Peak_Dict(origin_data, peak_low)
        length = len(peak_dict.keys())
        if length == temp_length:
            peak_value.append(peak[i])
            temp_length = len(peak_value) + 1
            center_record.append(center[i])
            coords_max = np.where(rotate_data == rotate_data.max())
            peak_location.append([coords_max[0][0], coords_max[1][0]])
            clump_size.append(sigma_t)
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

