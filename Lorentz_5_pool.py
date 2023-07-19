import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# from DataPreprocess import getZspec_49_middle, getZspec_56_middle
# from B0Correction import getB0

# from utils import plotZspecROI

def lorentz(x, amp, pos, width):
    tmp = ((x - pos) / width) * ((x - pos) / width)
    ret = amp / (1 + 4*tmp)
    return ret

def func_5pool(x, amp_noe, pos_noe, width_noe,
               amp_mt, pos_mt, width_mt,
               amp_ds, pos_ds, width_ds, 
               amp_cest, pos_cest, width_cest,
               amp_apt, pos_apt, width_apt):
    noe = lorentz(x, amp_noe, pos_noe, width_noe)
    mt = lorentz(x, amp_mt, pos_mt, width_mt)
    ds = lorentz(x, amp_ds, pos_ds, width_ds)
    cest = lorentz(x, amp_cest, pos_cest, width_cest)
    apt = lorentz(x, amp_apt, pos_apt, width_apt)
    return 100 - (noe + mt + ds + cest + apt)
    
def func_fixed_5pool(x, amp_noe, width_noe, amp_mt, width_mt, amp_ds, width_ds,
                     amp_cest, width_cest, amp_apt, width_apt):
    noe = lorentz(x, amp_noe, -3.5, width_noe)
    mt = lorentz(x, amp_mt, -2.34, width_mt)
    ds = lorentz(x, amp_ds, 0, width_ds)
    cest = lorentz(x, amp_cest, 2, width_cest)
    apt = lorentz(x, amp_apt, 3.5, width_apt)
    return 100 - (noe + mt + ds + cest + apt)
  
def calculateComponent(offset, paras):
    noe = []
    mt = []
    ds = []
    cest = []
    apt = []
    zspec = []
    for x in offset:
        noe.append(lorentz(x, *paras[0:3]))
        mt.append(lorentz(x, *paras[3:6]))
        ds.append(lorentz(x, *paras[6:9]))
        cest.append(lorentz(x, *paras[9:12]))
        apt.append(lorentz(x, *paras[12:15]))
        zspec.append(100 - (noe[-1]+mt[-1]+ds[-1]+cest[-1]+apt[-1]))

    return noe, mt, ds, cest, apt, zspec

def plot_5pool(offset, noe, mt, ds, cest, apt, zspec, title):
    plt.ylim((0, 100))
    plt.scatter(offset, noe, label='noe', color='green')
    plt.scatter(offset, mt, label='mt', color='purple')
    plt.scatter(offset, ds, label='ds', color='blue')
    plt.scatter(offset, cest, label='cest', color='yellow')
    plt.scatter(offset, apt, label='apt', color='red')
    plt.scatter(offset, zspec, label='zspec', color='black')
    plt.gca().invert_xaxis()  # invert x-axis
    plt.legend()
    plt.title(title)
    plt.show()    
    
def plot_fixed_5pool(offset, paras, title):
    noe_paras = paras[0:2]
    mt_paras = paras[2:4]
    ds_paras = paras[4:6]
    cest_paras = paras[6:8]
    apt_paras = paras[8:10]
    noe = []
    mt = []
    ds = []
    cest = []
    apt = []
    zspec = []
    for x in offset:
        noe.append(lorentz(x, noe_paras[0], -3.5, noe_paras[1]))
        mt.append(lorentz(x, mt_paras[0], -1.5, mt_paras[1]))
        ds.append(lorentz(x, ds_paras[0], 0, ds_paras[1]))
        cest.append(lorentz(x, cest_paras[0], 2, cest_paras[1]))
        apt.append(lorentz(x, apt_paras[0], 3.5, apt_paras[1]))
        zspec.append(100 - (noe[-1]+mt[-1]+ds[-1]+cest[-1]+apt[-1]))
    
    plt.scatter(offset, noe, label='noe', color='green')
    plt.scatter(offset, mt, label='mt', color='purple')
    plt.scatter(offset, ds, label='ds', color='blue')
    plt.scatter(offset, cest, label='cest', color='yellow')
    plt.scatter(offset, apt, label='apt', color='red')
    plt.scatter(offset, zspec, label='zspec', color='black')
    plt.gca().invert_xaxis()  # invert x-axis
    plt.legend()
    plt.title(title)
    plt.show()

def fitOnePixel(free, offset, zarray, method, initials, num_iter, bounds, title, b0_shift):
    popt = None
    popt_uncorrected = None
    if free:
        popt, pcov = curve_fit(func_5pool, xdata=offset, ydata=zarray, method=method,
                           p0=initials, maxfev=num_iter, bounds=bounds)
        popt_uncorrected = popt.copy()
        popt[1] -= b0_shift; popt[4] -= b0_shift; popt[7] -= b0_shift;
        popt[10] -= b0_shift; popt[13] -= b0_shift;
    else:
        popt, pcov = curve_fit(func_fixed_5pool, xdata=offset, ydata=zarray, method=method,
                           p0=initials, maxfev=num_iter, bounds=bounds)
        paras = np.zeros(15)
        paras[0] = popt[0]; paras[1] = -3.5 - b0_shift; paras[2] = popt[1];
        paras[3] = popt[2]; paras[4] = -2.34 - b0_shift; paras[5] = popt[3];
        paras[6] = popt[4]; paras[7] = 0 - b0_shift; paras[8] = popt[5];
        paras[9] = popt[6]; paras[10] = 2 - b0_shift; paras[11] = popt[7];
        paras[12] = popt[8]; paras[13] = 3.5 - b0_shift; paras[14] = popt[9];
        popt = paras
    # print("estimated parameters:", popt)
    # print(popt_uncorrected)
    # y_fitted = []
    # for x in offset:
    #     if free:
    #         y_fitted.append(func_5pool(x, *popt_uncorrected))
    #     else:
    #         y_fitted.append(func_fixed_5pool(x, *popt_uncorrected))
    # noe, mt, ds, cest, apt, zspec = calculateComponent(offset, popt_uncorrected)
    # plot_5pool(offset, noe, mt, ds, cest, apt, zarray, title)
    
    # print("y_fitted:", y_fitted)
    # plt.scatter(offset, zarray, label='y', color='blue')
    # plt.scatter(offset, y_fitted, label='y_fitted', color='red')
    # plt.gca().invert_xaxis()  # invert x-axis
    # plt.legend()
    # plt.title(title)
    # plt.show()   
    return popt

def fitROI(free, offset, zspec_arr, method, num_iter, patient, label):
    noe_fitted =[]; mt_fitted =[]; ds_fitted =[]; cest_fitted =[]; apt_fitted =[]
    zspec_fitted =[]
    paras_all = [] # store fitted parameters
    # get ROI info
    roi = pd.read_excel(r"C:\Users\jwu191\Desktop\mapping_output\ROI.xlsx")
    data = roi.loc[roi['name'] == patient]
    center_x = None; center_y = None; width = None; height = None; title = patient
    if label == 0:
        center_x = int(data['WM-X'] - 1)
        center_y = int(data['WM-Y'] - 1)
        width = height = int(data['WM-size'])
        title += '-WM'
    elif label == 1:
        center_x = int(data['GM-X'] - 1)
        center_y = int(data['GM-Y'] - 1)
        width = height = int(data['GM-size'])
        title += '-GM'
    elif label == 2:
        center_x = int(data['L-X'] - 1)
        center_y = int(data['L-Y'] - 1)
        width = height = int(data['L-size'])
        title += '-L'
    # get b0 map for correction
    wassr_path = os.path.join(r"C:\Users\jwu191\Desktop\mapping_output", patient, 
                              'WASSR-nifti', list(data['WASSR'])[0])
    for i in range(center_x - width, center_x + width + 1):
        for j in range(center_y - height, center_y + height + 1):
            # devide 128 for 3T
            b0_shift = getB0(wassr_path, i, j, int(data['n_slice'])) / 128
            # print('b0_shift:', b0_shift)
            # get initials
            if free:
                initials, bounds = ini_5pool_v2(b0_shift)
            else:
                initials, bounds = ini_fixed_5pool_v0()
            zarray = zspec_arr[:, i, j]
            paras = fitOnePixel(free, offset, zarray, method, initials, num_iter, 
                                bounds, title, b0_shift)
            paras_all.append(paras)
            noe, mt, ds, cest, apt, zspec = calculateComponent(offset, paras)
            noe_fitted.append(noe)
            mt_fitted.append(mt)
            ds_fitted.append(ds)
            cest_fitted.append(cest)
            apt_fitted.append(apt)
            zspec_fitted.append(zspec)
    noe_avg = np.mean(np.array(noe_fitted), axis=0)
    mt_avg = np.mean(np.array(mt_fitted), axis=0)
    ds_avg = np.mean(np.array(ds_fitted), axis=0)
    cest_avg = np.mean(np.array(cest_fitted), axis=0)
    apt_avg = np.mean(np.array(apt_fitted), axis=0)
    zspec_avg = np.mean(np.array(zspec_fitted), axis=0)

    plot_5pool(offset, noe_avg, mt_avg, ds_avg, cest_avg, apt_avg, zspec_avg, title) 
    # print(np.array(paras_all).shape)
    column_names = ["NOE-amp", "NOE-pos", "NOE-width", "MT-amp", "MT-pos", "MT-width",
             "DS-amp", "DS-pos", "DS-width", "CEST-amp", "CEST-pos", "CEST-width",
             "APT-amp", "APT-pos", "APT-width"]
    data = np.mean(np.array(paras_all), axis=0) # len:15
    return data
    
def fitByPatient(free, patient):
    # get Zspec, scanned offset
    zspectra_all, scanned_offset = getZspec_56_middle(patient)
    print(zspectra_all.shape)
    print(scanned_offset)
    # plot Zspec
    plotZspecROI(patient, zspectra_all, scanned_offset)
    # fit WM, GM, L (ROI)
    data_wm = fitROI(free=free, offset=scanned_offset, zspec_arr=zspectra_all[0],
                method='trf', num_iter=5000, patient=patient, label=0)

    data_gm = fitROI(free=free, offset=scanned_offset, zspec_arr=zspectra_all[0],
                method='trf', num_iter=5000, patient=patient, label=1)

    data_l = fitROI(free=free, offset=scanned_offset, zspec_arr=zspectra_all[0],
                method='trf', num_iter=5000, patient=patient, label=2)
    data = np.vstack((data_wm, data_gm, data_l))
    # print(data.shape)
    column_names = ["NOE-amp", "NOE-pos", "NOE-width", "MT-amp", "MT-pos", "MT-width",
             "DS-amp", "DS-pos", "DS-width", "CEST-amp", "CEST-pos", "CEST-width",
             "APT-amp", "APT-pos", "APT-width"]
    names = [patient+ '-WM', patient+ '-GM', patient+ '-L']
    df_result = pd.DataFrame(data=data, index=names, columns=column_names)
    save_path = os.path.join(r"C:\Users\jwu191\Desktop\my_lorentz\5 pool results", patient+'.csv')
    df_result.to_csv(save_path)
    return data

def combineResult(patients):
    index = []
    wm_data = []
    gm_data = []
    l_data = []
    for patient in patients:
        path = os.path.join(r"C:\Users\jwu191\Desktop\my_lorentz\5 pool results", patient+'.csv')
        df = pd.read_csv(path)
        wm_data.append(np.array(df.iloc[0][1:]))
        gm_data.append(np.array(df.iloc[1][1:]))
        l_data.append(np.array(df.iloc[2][1:]))  
    wm_avg = np.mean(np.array(wm_data), axis=0)
    gm_avg = np.mean(np.array(gm_data), axis=0)
    l_avg = np.mean(np.array(l_data), axis=0)
    data = np.vstack((np.array(wm_data), np.array(gm_data), np.array(l_data),
                     wm_avg, gm_avg, l_avg))
    names = ['-WM', '-GM', '-L']
    for i in range(3):
        for patient in patients:
            index.append(patient+names[i])
    index += ['WM-AVG', 'GM-AVG', 'L-AVG']    
    column_names = ["NOE-amp", "NOE-pos", "NOE-width", "MT-amp", "MT-pos", "MT-width",
             "DS-amp", "DS-pos", "DS-width", "CEST-amp", "CEST-pos", "CEST-width",
             "APT-amp", "APT-pos", "APT-width"]
    df_result = pd.DataFrame(data=data, index=index, columns=column_names)
    df_result.to_csv(r"C:\Users\jwu191\Desktop\my_lorentz\5 pool results\result.csv")
    return





