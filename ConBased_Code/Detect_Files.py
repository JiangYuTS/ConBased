#usr JiangYu
from ConBased.Detect_Files_Funs import Detect

if __name__ == '__main__':
    #2D,3D
    RMS = 0.23
    Threshold = 2 * RMS  # ['mean','otsu',n*RMS]
    RegionMin = [27] # [9,18,27],[27,64,125]
    ClumpMin = [216] # [27,...]，[125,...]
    DIntensity = 2 * RMS # [-3*RMS,3*RMS]
    DDistance = [8] # [4,16]

    # RegionMin_LBV = [3 * 3, 3]
    # ClumpMin_LBV = [6 * 6, 6]
    # DDistance_LBV = [6, 6]

    parameters = [RMS, Threshold, RegionMin, ClumpMin, DIntensity, DDistance]
    file_name = 'file_name'
    mask_name = 'mask.fits'
    outcat_name = 'outcat.csv'
    outcat_wcs_name = 'outcat_wcs.csv'
    did_ConBased = Detect(file_name, parameters, mask_name, outcat_name, outcat_wcs_name)
