#usr JiangYu

from ConBased.ConBased_2D_Funs import Detect_ConBased as Detect_ConBased_2D
from ConBased.ConBased_3D_Funs import Detect_ConBased as Detect_ConBased_3D
from ConBased.ConBased_3D_Funs_LBV import Detect_ConBased_LBV as Detect_ConBased_3D_LBV


if __name__ == '__main__':
    origin_data = 'origin_data'
    RMS = 0.23
    Threshold = 2 * RMS  # ['mean','otsu',n*RMS]
    DIntensity = 2 * RMS  # [-3*RMS,3*RMS]

    #2D
    RegionMin = 9  # [9,18,27]
    ClumpMin = 27  # [27,...]
    DDistance = 6  # [4,16]
    did_ConBased = Detect_ConBased_2D(RMS, Threshold, RegionMin, ClumpMin, DIntensity, DDistance, origin_data)
    #3D
    RegionMin = 27  # [27,64,125]
    ClumpMin = 216  # [125,...]
    DDistance = 8  # [6,16]
    did_ConBased = Detect_ConBased_3D(RMS, Threshold, RegionMin, ClumpMin, DIntensity, DDistance, origin_data)
    #3D LBV
    RegionMin_LBV = [3 * 3, 3]
    ClumpMin_LBV = [6 * 6, 6]
    DDistance_LBV = [6, 6]
    did_ConBased_LBV = Detect_ConBased_3D_LBV(RMS, Threshold, RegionMin_LBV, ClumpMin_LBV, DIntensity, DDistance_LBV, origin_data)