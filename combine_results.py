import numpy as np
import pandas as pd
import os


def combine_Lorentz(patient_list, roi):
    result = []
    column_names = ["NOE-amp", "NOE-pos", "NOE-width", "MT-amp", "MT-pos", "MT-width",
                    "DS-amp", "DS-pos", "DS-width", "CEST-amp", "CEST-pos", "CEST-width",
                    "APT-amp", "APT-pos", "APT-width", "B0 shift"]
    for patient_name in patient_list:
        df = pd.read_csv(os.path.join(results_dir, patient_name, patient_name+"_"+roi+"_Lorentz.csv"))
        data = np.array(df)[:, 1:]
        item = np.mean(data, axis=0)
        result.append(item)
    df_result = pd.DataFrame(data=np.array(result), index=None, columns=column_names)
    df_result.to_csv(r"C:\Users\jwu191\Desktop\Lorentz_"+roi+".csv")

def combine_EMR(patient_name, roi):
    result = []
    column_names = ["R", "R*M0m*T1", "T1w/T2w", "T2m", "APT_pow", "MTR_asym", "residual"]
    for patient_name in patient_list:
        df = pd.read_csv(os.path.join(results_dir, patient_name, patient_name+"_"+roi+"_EMR.csv"))
        data = np.array(df)[:, 1:]
        item = np.mean(data, axis=0)
        result.append(item)
    df_result = pd.DataFrame(data=np.array(result), index=None, columns=column_names)
    df_result.to_csv(r"C:\Users\jwu191\Desktop\EMR_"+roi+".csv")

results_dir = r"C:\Users\jwu191\Desktop\EMR fitting\results"
patient_list = ["APT_475", "APT_476", "APT_477", "APT_478", "APT_479", "APT_482", "APT_483",
                "APT_486", "APT_487", "APT_488", "APT_489", "APT_491", "APT_492", "APT_493"]    
roi = "tumor"

combine_Lorentz(patient_list, roi)
combine_EMR(patient_list, roi)





