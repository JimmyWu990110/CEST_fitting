import os
import numpy as np
import pandas as pd

from DataLoader import *
from Visualization import plot_Zspec, plot_fitting_result
from Initialization import ini_5pool_v2
from B0Correction import getB0
from EMR_fitting import *
from Lorentz_5_pool import fitOnePixel

mapping_dir = r"C:\Users\jwu191\Desktop\mapping_output"
output_dir = r"C:\Users\jwu191\Desktop\output"
patient_name = "Phantom_20230503b"
nth_slice = 8 # the slice selected for Zspec
phantom_label = 3
seq_num = 17 # Zspec id

B0 = 3
B1 = 1.5
gyr = 42.576
    
scanned_offset_ppm = get_Zspec_56_offsets_ppm()
# selected_offset_ppm = get_doubleside_wide_14_offsets_ppm()    
selected_offset_ppm = get_middle_41_offsets_ppm()
   
roi = get_phantom_mask(patient_name, nth_slice, phantom_label) # size: ~ 700    
Zspec_array = read_Zspec_by_id(os.path.join(mapping_dir, patient_name), seq_num, nth_slice)   
# print(Zspec_array.shape) # (56, 256, 256)   

def fit_Lorentz():
    data = []
    print("size: ", roi[0].shape[0])
    for i in range(roi[0].shape[0]):
        Zspec_onepixel = Zspec_array[:, roi[0][i], roi[1][i]]
        Zspec, offset_ppm = Zspec_preprocess_onepixel(Zspec_onepixel, scanned_offset_ppm, 
                                                      selected_offset_ppm)
        # plot_Zspec(offset_ppm, Zspec, str(i+1))
        WASSR_path = os.path.join(mapping_dir, patient_name, "WASSR-nifti", 
                                  "Phantom_20230503b_WIPAxial_WASSR_15sl_20230503095927_16.nii")
        b0_shift = getB0(WASSR_path, roi[0][i], roi[1][i], nth_slice)
        print(i, "b0_shift:", b0_shift)
        # Lorentz fitting
        initials, bounds = ini_5pool_v2(b0_shift/128)
        fitted_paras = fitOnePixel(free=True, offset=offset_ppm, zarray=100*Zspec, method="trf", initials=initials, 
                    num_iter=5000, bounds=bounds, title=str(i+1), b0_shift=b0_shift/128)
        data.append(list(fitted_paras) + [b0_shift])
    # save results    
    column_names = ["NOE-amp", "NOE-pos", "NOE-width", "MT-amp", "MT-pos", "MT-width",
                    "DS-amp", "DS-pos", "DS-width", "CEST-amp", "CEST-pos", "CEST-width",
                    "APT-amp", "APT-pos", "APT-width", "B0 shift"]
    data = np.array(data) # len:16
    df_result = pd.DataFrame(data=data, index=None, columns=column_names)
    save_path = os.path.join(r"C:\Users\jwu191\Desktop", patient_name+"_ROI_"+str(phantom_label)+".csv")
    df_result.to_csv(save_path)

def fit_EMR():
    data = []
    print("size: ", roi[0].shape[0])
    for i in range(roi[0].shape[0]):
        Zspec_onepixel = Zspec_array[:, roi[0][i], roi[1][i]]
        Zspec, offset_ppm = Zspec_preprocess_onepixel(Zspec_onepixel, scanned_offset_ppm, 
                                                      selected_offset_ppm)
        frequencies = offset_ppm * B0 * gyr
        is_constrained = True
        # [R, R*M0m*T1w, T1w/T2w, T2m]
        y_estimated, fitted_paras = FitMTmodel_onepixel(frequencies, Zspec, B1, 40, 1, 
                                                              is_constrained)
        plot_fitting_result(offset_ppm, y_estimated, Zspec, str(i+1), 3.5)
        diff = 100*(Zspec-y_estimated)
        APT_pow = np.mean(diff[np.where(offset_ppm == 3.5)]) # array with only 1 element
        residual = np.linalg.norm(diff, ord=2) / Zspec.shape[0]
        # print(i, fitted_parameters, residual)
        data.append(list(fitted_paras) + [APT_pow, residual])
    column_names = ["R", "R*M0m*T1", "T1w/T2w", "T2m", "APT_pow", "residual"]
    df_result = pd.DataFrame(data=data, index=None, columns=column_names)
    save_path = os.path.join(r"C:\Users\jwu191\Desktop", "EMR_"+patient_name+"_ROI_"+str(phantom_label)+".csv")
    df_result.to_csv(save_path)

# fit_Lorentz()
fit_EMR()

    





        
        