#usr JiangYu
import numpy as np
from ConBased.Generate_2D_Funs import Generate_2D
from ConBased.Generate_3D_Funs import Generate_3D

if __name__ == '__main__':
    n = 100
    peak_low = 0.69
    peak_high = 4.6
    angle = None
    rms = 0.23
    xres = 100
    yres = 100
    zres = 100
    sigma_one = [2, 4]
    sigma_two = [2, 4]
    sigma_three = [2, 4]
    ctime = np.sqrt(8 * np.log(2))

    origin_data_2D,noise_data_2D,infor_dict_2D = Generate_2D(n,xres,yres,peak_low,peak_high,ctime,sigma_one,sigma_two,angle)
    origin_data_3D,noise_data_3D,infor_dict_3D = Generate_3D(n,xres,yres,zres,peak_low,peak_high,ctime,sigma_one,sigma_two,
                                                      sigma_three,rms,angle)
