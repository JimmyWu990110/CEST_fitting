import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from EMR_fitting import EMR_fitting
from DataLoader import DataLoader
from Visualization import Visualization

def offset_preprocess(offset):
    offset = np.array(offset)
    sorted_index = np.argsort(offset)[::-1]
    return offset[sorted_index]

def get_avg_Zspec(path):
    df = pd.read_csv(path) 
    data = np.array(df)
    data = data[:, 3:]
    return np.mean(data, axis=0)

class Simulation:
    def __init__(self, B1):
        self.B1 = B1
        self.wide_freq = 128 * offset_preprocess(DataLoader.Zspec_offset.wide_14)
        self.middle_freq = 128 * offset_preprocess(DataLoader.Zspec_offset.middle_41)
        self.Zspec_wide = None
        self.Zspec_middle = None
        
    @staticmethod
    def plotZspec(offset, Zspec, title):
        plt.ylim((0, 1))
        plt.scatter(offset, Zspec, label='Zspec', color='black')
        plt.gca().invert_xaxis()  # invert x-axis
        plt.legend()
        plt.title(title)
        plt.show() 
        
    @staticmethod       
    def compareZspec(offset, Zspec_simulated, Zspec_patient, title):
        plt.ylim((0, 1))
        plt.scatter(offset, Zspec_patient, label='Zspec_patient', color='green')
        plt.scatter(offset, Zspec_simulated, label='Zspec_simulated', color='red')
        plt.gca().invert_xaxis()  # invert x-axis
        plt.legend()
        plt.title(title)
        plt.show() 
        
    def simulate(self, paras):
        fitting = EMR_fitting(1.5, 1, 0.1)
        Zspec_wide = fitting.MT_model(self.wide_freq, *paras)
        Zspec_middle = fitting.MT_model(self.middle_freq, *paras)
        # self.plotZspec(self.wide_freq/128, self.Zspec_wide, str(paras))
        # self.plotZspec(self.middle_freq/128, self.Zspec_middle, str(paras))
        return Zspec_wide, Zspec_middle

    def load_patient(self, patient_name):
        dataloader = DataLoader(patient_name)
        dataloader.get_nth_slice_by_Zspec_mask()
        dataloader.get_ROIs_by_Zspec_mask()
        dataloader.read_Zspec(self.B1)
        Zspec_wide_patient = dataloader.get_tumor_avg_Zspec(self.wide_freq/128)
        self.compareZspec(self.wide_freq/128, self.Zspec_wide, 
                          Zspec_wide_patient, "wide")
        Zspec_middle_patient = dataloader.get_tumor_avg_Zspec(self.middle_freq/128)
        self.compareZspec(self.middle_freq/128, self.Zspec_middle, 
                          Zspec_middle_patient, "wide")
        
    def evaluate_R(self):
        paras = np.array([[2, 0.2, 10, 8e-6],
                          [20, 2, 10, 8e-6],
                          [200, 20, 10, 8e-6],
                          [2000, 200, 10, 8e-6],
                          [20000, 2000, 10, 8e-6],
                          [200000, 20000, 10, 8e-6]])
        colors = ["red", "orange", "yellow", "green", "blue", "purple"]
        for i in range(paras.shape[0]):
            Zspec_wide, Zspec_middle = self.simulate(paras[i])
            plt.scatter(self.wide_freq/128, Zspec_wide, color=colors[i])
            # plt.scatter(self.middle_freq/128, Zspec_middle, color=colors[i])
        plt.show()


# [R, R_M0m_T1w, T1w_T2w, T2m]
# paras = np.array([[10, 1, 40, 8e-6],
#                   [50, 5, 40, 8e-6],
#                   [500, 50, 40, 8e-6],
#                   [5000, 500, 40, 8e-6],
#                   [50000, 5000, 40, 8e-6]])

obj = Simulation(B1=1.5)
# obj.simulate([50, 5, 40, 8e-6])
# obj.simulate([20, 2, 10, 8e-6])
obj.evaluate_R()


# Visualization.plot_2_Zspec(obj.wide_freq/128, Zspec_1_wide, Zspec_2_wide)
# Visualization.plot_2_Zspec(obj.middle_freq/128, Zspec_1_middle, Zspec_2_middle)
# obj.load_patient("APT_479")

# plt.scatter(wide_freq/128, Zspec_wide, label='Zspec_wide', color='blue') 
# plt.scatter(middle_freq/128, Zspec_middle, label='Zspec_middle', color='green') 
# plt.scatter(frequencies / (B0*42.576), Zspec_2, label='Zspec_2', color='yellow') 
# plt.scatter(frequencies / (B0*42.576), Zspec_3, label='Zspec_3', color='purple') 
# plt.scatter(frequencies / (B0*42.576), Zspec_4, label='Zspec_4', color='black') 
 
# Zspec = get_avg_Zspec(r"C:\Users\jwu191\Desktop\EMR fitting\Zspec_1.5uT_56offsets\20220720_JG_normal_Zspec.csv")
# plt.scatter(frequencies / (B0*42.576), Zspec/100, label='case', color='red')
# plt.title("20220720_JG_normal")   
# plt.legend()  
# plt.show()
   
        
        
        