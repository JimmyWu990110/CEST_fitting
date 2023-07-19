import os

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from DataLoader import DataLoader
from Initialization import Initialization
from Lorentz_fitting import Lorentz_fitting
from EMR_fitting import EMR_fitting
from Correction import B0_Correction
from Visualization import Visualization


def fit_Lorentz(patient_name, roi_type):
    visualization = Visualization()
    dataloader = DataLoader(patient_name)
    WASSR = dataloader.get_WASSR_Zspec()
    B0_cor = B0_Correction(WASSR)
    roi = None
    if roi_type == "normal":
        tumor, normal = dataloader.get_ROIs_by_Zspec_mask()
        roi = normal
    elif roi_type == "tumor":
        tumor, normal = dataloader.get_ROIs_by_Zspec_mask()
        roi = tumor
    elif roi_type.startswith("phantom-"):
        dataloader.nth_slice = 8
        label = int(roi_type.split("-")[-1])
        roi = dataloader.get_ROIs_by_phantom_mask(label)
    else:
        raise Exception("Invalid Zspec mask!")
    filename = "Lorentz-" + patient_name + "-" + roi_type
    data = []
    for i in range(roi[0].shape[0]):
        print("******** pixel", i, "********")
        Zspec_raw = dataloader.Zspec_img[:, roi[0][i], roi[1][i]]
        Zspec, offset = dataloader.Zspec_preprocess_onepixel(Zspec_raw, 
                                                             DataLoader.Zspec_offset.full_56, 
                                                             DataLoader.Zspec_offset.middle_41) 
        B0_shift_hz = B0_cor.cal_B0_shift_onepixel(dataloader.nth_slice, roi[0][i], roi[1][i])
        print("B0 shift:", B0_shift_hz, "hz")
        B0_shift_ppm = B0_shift_hz / 128
        # Visualization.plot_Zspec(offset, Zspec, filename+"-"+str(i))
        # Lorentz fitting
        initialization = Initialization(B0_shift_ppm)
        initialization.init_2_pool() # 2
        fitting = Lorentz_fitting(initialization)
        fitted_paras = fitting.fit_2_pool(offset, Zspec)
        y_estimated = fitting.generate_Zpsec("2_pool", offset, fitted_paras)
        print("fitted parameters:", fitted_paras)
        visualization.plot_2_Zspec(offset, Zspec, y_estimated, ["real", "fitted"],
                                   filename+"-"+str(i), 3.5)
        visualization.plot_component(offset, fitted_paras, filename+"-"+str(i))
        # extraplot to all offset
        Zspec_all, offset_all = dataloader.Zspec_preprocess_onepixel(Zspec_raw, 
                                                                     DataLoader.Zspec_offset.full_56, 
                                                                     DataLoader.Zspec_offset.full_55)
        Zspec_all_extrap = fitting.generate_Zpsec("2_pool", offset_all, fitted_paras)
        visualization.plot_2_Zspec(offset_all, Zspec_all, Zspec_all_extrap, ["real", "fitted"],
                                    filename+"-"+str(i), 3.5)
        visualization.plot_component(offset_all, fitted_paras, filename+"-"+str(i))
        # add results
        data.append(list(fitted_paras) + [B0_shift_hz])        
    # save results    
    column_names = ["MT-amp", "MT-pos", "MT-width", "DS-amp", "DS-pos", "DS-width", "B0 shift"]
    data = np.array(data)
    df_result = pd.DataFrame(data=data, index=None, columns=column_names)
    out_dir = os.path.join(r"C:\Users\jwu191\Desktop")
    # # os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, filename+".csv")
    df_result.to_csv(save_path)

def fit_EMR(patient_name, roi_type, model_type, is_constrained):
    visualization = Visualization()
    dataloader = DataLoader(patient_name)
    roi = None
    if roi_type == "normal":
        tumor, normal = dataloader.get_ROIs_by_Zspec_mask()
        roi = normal
    elif roi_type == "tumor":
        tumor, normal = dataloader.get_ROIs_by_Zspec_mask()
        roi = tumor
    elif roi_type.startswith("phantom-"):
        dataloader.nth_slice = 8
        label = int(roi_type.split("-")[-1])
        roi = dataloader.get_ROIs_by_phantom_mask(label)
    else:
        raise Exception("Invalid Zspec mask!")
    filename = "EMR-model-" + str(model_type) + "-" + patient_name + "-" + roi_type
    if is_constrained:
        filename += "-constrained"
    else:
        filename += "-unconstrained"
    data = []
    for i in range(roi[0].shape[0]):
        print("******** pixel", i, "********")
        Zspec_raw = dataloader.Zspec_img[:, roi[0][i], roi[1][i]]
        T1w_obs = dataloader.T1_map[roi[0][i], roi[1][i]]
        T2w_obs = dataloader.T2_map[roi[0][i], roi[1][i]]
        print("T1 map, T2 map:", T1w_obs, T2w_obs)
        print("T1/T2:", T1w_obs/T2w_obs)
        Zspec, offset = dataloader.Zspec_preprocess_onepixel(Zspec_raw, 
                                                             DataLoader.Zspec_offset.full_56, 
                                                             DataLoader.Zspec_offset.combine_7_1)                                                      
        freq = offset * 128 # offset in ppm -> frequency in hz
        # initialization
        fitting = EMR_fitting(T1w_obs=T1w_obs, T2w_obs=T2w_obs)
        fitting.set_model_type(model_type)
        initialization = Initialization(0)
        initialization.init_EMR(T1w_obs/T2w_obs, model_type)
        fitting.set_x0(initialization.x0)
        fitting.set_lb(initialization.lb)
        fitting.set_ub(initialization.ub)
        # fit the data 
        fitted_paras, y_estimated = fitting.fit(freq, Zspec, is_constrained)
        all_paras = fitting.cal_paras()
        print("All parameters:\n", all_paras)
        visualization.plot_2_Zspec(offset, Zspec, y_estimated, ["real", "fitted"],
                                   filename+"-"+str(i), 3.5)
        # change lineshape when extraplot to middle offset
        # fitting.set_lineshape("L")
        # extraplot to all offset
        Zspec_all, offset_all = dataloader.Zspec_preprocess_onepixel(Zspec_raw, 
                                                                     DataLoader.Zspec_offset.full_56, 
                                                                     DataLoader.Zspec_offset.full_55)
        Zspec_all_extrap = fitting.generate_Zpsec(128*offset_all, fitted_paras)
        visualization.plot_2_Zspec(offset_all, Zspec_all, Zspec_all_extrap, ["real", "fitted"],
                                    filename+"-"+str(i), 3.5)
        # extraplot to middle offset
        Zspec_middle, offset_middle = dataloader.Zspec_preprocess_onepixel(Zspec_raw, 
                                                                           DataLoader.Zspec_offset.full_56,
                                                                           DataLoader.Zspec_offset.middle_45)
        Zspec_middle_extrap = fitting.generate_Zpsec(128*offset_middle, fitted_paras)
        visualization.plot_2_Zspec(offset_middle, Zspec_middle, Zspec_middle_extrap, ["real", "fitted"],
                                   filename+"-"+str(i), 3.5)
        # calculate MTR_asym, APT# and residuals
        MTR_asym = (Zspec_middle[np.where(offset_middle == -3.5)[0][0]] - 
                    Zspec_middle[np.where(offset_middle == 3.5)[0][0]])
        diff = 100 * (y_estimated - Zspec)
        diff_middle = 100 * (Zspec_middle_extrap - Zspec_middle)
        diff_all = 100 * (Zspec_all_extrap - Zspec_all)
        APT_pow = np.mean(diff_middle[np.where(offset_middle == 3.5)])
        residual_fitting = np.linalg.norm(diff, ord=2) / Zspec.shape[0]
        residual_extrap = np.linalg.norm(diff_middle, ord=2) / Zspec_middle.shape[0]
        r2_fitting = r2_score(Zspec, y_estimated)
        r2_extrap = r2_score(Zspec_middle, Zspec_middle_extrap)
        print("MTR_asym, APT#:", MTR_asym, APT_pow)
        print("residuals (fitted and extraploted):", residual_fitting, residual_extrap)
        print("r2 (fitted and extraploted):", r2_fitting, r2_extrap)
        visualization.plot_APT(offset_middle, diff_middle, filename+"-"+str(i))
        # visualization.plot_APT(offset_all, diff_all, filename+"-"+str(i))
        # add results
        data.append(list(all_paras.values()) + 
                    [T1w_obs, T2w_obs, T1w_obs/T2w_obs, APT_pow, MTR_asym, 
                     residual_fitting, residual_extrap, r2_fitting, r2_extrap])
    # save results
    column_names = ["R", "R*M0m*T1w", "T1w/T2w", "T2m", "M0m", "T1w", "T2w",
                    "T1w_obs", "T2w_obs", "T1w_obs/T2w_obs", "APT_pow", "MTR_asym", 
                    "residual_fitting", "residual_extrap", "r2_fitting", "r2_extrap"]
    data = np.array(data)
    # calculate avg
    tmp = np.mean(data, axis=0)
    for i, col in enumerate(column_names):
        print(col+":", tmp[i])
        
    df_result = pd.DataFrame(data=data, index=None, columns=column_names)
    out_dir = os.path.join(r"C:\Users\jwu191\Desktop")
    # # os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, filename+".csv")
    df_result.to_csv(save_path)


# mapping_dir = r"C:\Users\jwu191\Desktop\mapping_output"
# processed_cases_dir = r"C:\Users\jwu191\Desktop\processed_cases"
# mask_dir = r"C:\Users\jwu191\Desktop\EMR fitting\Zspec_mask"
patient_name = "APT_479"

# fit_Lorentz(patient_name, "tumor")
# fit_Lorentz(patient_name, "normal")
# fit_Lorentz(patient_name, "phantom-1")

fit_EMR(patient_name=patient_name, roi_type="tumor", model_type=2, is_constrained=False)

    

# def fit_EMR(patient_name, roi, filename=""):
#     data = []
#     print("size: ", roi[0].shape[0])
#     for i in range(roi[0].shape[0]):
#         Zspec_onepixel = Zspec_array[:, roi[0][i], roi[1][i]]
#         Zspec, offset_ppm = Zspec_preprocess_onepixel(Zspec_onepixel, scanned_offset_ppm, 
#                                                       selected_offset_ppm)
#         offset_hz = offset_ppm * 128
#         WASSR_path = get_WASSR_Zspec_path(processed_cases_dir, patient_name)
#         b0_shift = getB0(WASSR_path, roi[0][i], roi[1][i], nth_slice)
#         # B0 correction
#         Zspec_corrected = interpolate_B0_shift(offset_hz, Zspec, b0_shift)
#         MTR_asym = Zspec_corrected[np.where(offset_ppm == -3.5)[0][0]] - Zspec_corrected[np.where(offset_ppm == 3.5)[0][0]]
#         # frequencies = offset_ppm * B0 * gyr
#         is_constrained = True
#         # [R, R*M0m*T1w, T1w/T2w, T2m]
#         y_estimated, fitted_paras = FitMTmodel_onepixel(offset_hz, Zspec_corrected, B1, 40, 1, 
#                                                               is_constrained)
#         plot_fitting_result(offset_ppm, y_estimated, Zspec_corrected, filename+"-"+str(i), 3.5)
#         diff = 100 * (y_estimated-Zspec_corrected)
#         APT_pow = np.mean(diff[np.where(offset_ppm == 3.5)]) # array with only 1 element
#         residual = np.linalg.norm(diff, ord=2) / Zspec_corrected.shape[0]
#         print("MTR_asym:", MTR_asym, "APT#:", APT_pow)
#         # print(i, fitted_parameters, residual)
#         data.append(list(fitted_paras) + [APT_pow, MTR_asym, residual])
        
#     column_names = ["R", "R*M0m*T1", "T1w/T2w", "T2m", "APT_pow", "MTR_asym", "residual"]
#     df_result = pd.DataFrame(data=data, index=None, columns=column_names)
#     if len(filename) > 0:   
#         out_dir = os.path.join(r"C:\Users\jwu191\Desktop\EMR fitting\results", patient_name)
#         os.makedirs(out_dir, exist_ok=True)
#         save_path = os.path.join(out_dir, filename+"_EMR.csv")
#         df_result.to_csv(save_path)




        
        