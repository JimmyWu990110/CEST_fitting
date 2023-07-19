import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import SimpleITK as sitk

class Visualization:
    def __init__(self):
        pass
    
    def plot_Zspec(self, offset, Zspec, label="Zspec", title=""):
        """
        Plot a single Zspec given the offsets
        Zspec must be normalized to [0, 1]
        """
        plt.ylim((0, 1))
        plt.scatter(offset, Zspec, label=label, color="blue")
        plt.gca().invert_xaxis()  # invert x-axis
        plt.legend()
        plt.title(title)
        plt.show()
    
    def plot_2_Zspec(self, offset, Zspec_1, Zspec_2, labels=["Zspec_1","Zspec_2"], title="", highlight=None):
        plt.ylim((0, 1))
        plt.scatter(offset, Zspec_1, label=labels[0], color='blue')
        plt.scatter(offset, Zspec_2, label=labels[1], color='orange')
        # index = np.where(x==3.5)
        # plt.scatter(x[index], y_estimated[index], color='red')
        plt.gca().invert_xaxis()  # invert x-axis
        plt.legend()
        plt.title(title)
        plt.show() 
    
    def _lorentz(self, x, amp, pos, width):
        factor = ((x - pos) / width) * ((x - pos) / width)
        return amp / (1 + 4*factor)
    
    def plot_component(self, offset, paras, title=""):
        plt.ylim((0, 1))
        mt = []
        ds = []
        for x in offset:
            mt.append(self._lorentz(x, *paras[0:3]))
            ds.append(self._lorentz(x, *paras[3:]))
        plt.scatter(offset, np.array(mt), label="MT", color="purple")
        plt.scatter(offset, np.array(ds), label="DS", color="blue")
        plt.gca().invert_xaxis()  # invert x-axis
        plt.legend()
        plt.title(title)
        plt.show()
        
    def plot_APT(self, offset, y, title=""):
        plt.plot(offset, y)
        plt.scatter(offset, y)
        plt.gca().invert_xaxis()  # invert x-axis
        plt.title(title)
        plt.show()
        
    
# def plot_Zspec_diff(x, diff, title):
#     plt.scatter(x, diff, label='diff', color='blue')
#     index = np.where(x==3.5)
#     plt.scatter(x[index], diff[index], color='red')
#     plt.gca().invert_xaxis()  # invert x-axis
#     plt.legend()
#     plt.title(title)
#     plt.show() 

# def get_tumor_mask(output_dir, patient_name, nthslice):
#     roi_path = os.path.join(output_dir, patient_name+'_all', patient_name, 
#                         '4_coreg2apt', patient_name, patient_name+'_mask.nii.gz')
#     # ROI mask including edema, necrosis and tumor
#     mask = sitk.ReadImage(roi_path, sitk.sitkInt32)
#     # For Zspec, only use one specific slice
#     mask_array = sitk.GetArrayFromImage(mask)[nthslice-1]
#     size = np.sum(mask_array == 7)
#     print(patient_name, 'tumor size:', size)
#     roi = np.where(mask_array == 7)
#     return roi, size
    
# def get_normal_mask(output_dir, patient_name, nthslice):
#     roi_path = os.path.join(output_dir, patient_name+'_all', patient_name, 
#                         '4_coreg2apt', patient_name, patient_name+'_normal.nii.gz')
#     # ROI mask including edema, necrosis and tumor
#     mask = sitk.ReadImage(roi_path, sitk.sitkInt32)
#     # For Zspec, only use one specific slice
#     mask_array = sitk.GetArrayFromImage(mask)[nthslice-1]
#     size = np.sum(mask_array == 1)
#     print(patient_name, 'normal size:', size)
#     roi = np.where(mask_array == 1)
#     return roi, size

# def get_avg_Zspec(path):
#     df = pd.read_csv(path) 
#     data = np.array(df)
#     data = data[:, 3:]
#     return np.mean(data, axis=0)

# def read_lookup_table(lut_dir):
#     lookup_table = []
#     with open(lut_dir, "r") as lut_file:
#         lut_lines = lut_file.read().split('\n')
#         for line in lut_lines:
#             if len(line) > 0:
#                 line_nums = [int(i) for i in line.split('\t')]
#                 lookup_table.append(line_nums)
#     # 4 columns: Gray - R - G - B
#     lookup_table = np.array(lookup_table) 
#     return lookup_table

# def gray_to_idl(gray_img, lookup_table):
#     print("Min:", np.min(gray_img))
#     print("Max:", np.max(gray_img))
#     for i in range(gray_img.shape[0]):
#         for j in range(gray_img.shape[1]):
#             g_val = gray_img[i,j,2]
#             # blue channel
#             gray_img[i,j,0] = lookup_table[g_val,3]
#             # green channel
#             gray_img[i,j,1] = lookup_table[g_val,2]
#             gray_img[i,j,2] = lookup_table[g_val,1]
#     return gray_img

# mapping_dir = r"C:\Users\jwu191\Desktop\mapping_output"
# processed_cases_dir = r"C:\Users\jwu191\Desktop\processed_cases"
# output_dir = r"C:\Users\jwu191\Desktop\output"
# patient_name = "Phantom_321"
# nth_slice = 8
# phantom_label = 1
# seq_num = 18

# Visualize APTw
# roi = get_phantom_mask_all(nth_slice)
# img_path = os.path.join(processed_cases_dir, patient_name+"_all", patient_name,
#                         "4_coreg2apt", patient_name, patient_name+"_11_apt.nii")
# img = sitk.ReadImage(img_path, sitk.sitkInt32)
# arr = sitk.GetArrayFromImage(img)[nth_slice-1]
# arr[roi] = -5
# arr[0][0] = 5
# arr = np.flip(arr, 0) # axis=0 !
# arr = np.interp(arr, (arr.min(), arr.max()), (0, 255))
# arr[0][0] = 0
# cv2.imwrite(r"C:\Users\jwu191\Desktop\APTw.png", arr)  
# arr = cv2.imread(r"C:\Users\jwu191\Desktop\APTw.png", 1)
# lookup_table = read_lookup_table(r"C:\Users\jwu191\Desktop\my_pipeline\idl_rainbow.lut")
# arr = gray_to_idl(arr, lookup_table)
# cv2.imwrite(r"C:\Users\jwu191\Desktop\APTw.png", arr) 

# Visualize all ROIs
# def v_all():
#     mask_path = os.path.join(processed_cases_dir, patient_name+"_all", patient_name, 
#                               "4_coreg2apt", patient_name, patient_name+"_mask.nii.gz")
#     mask = sitk.ReadImage(mask_path, sitk.sitkInt32)
#     mask_arr = sitk.GetArrayFromImage(mask)[nth_slice-1]
#     img = np.ones((256, 256)) * -5
#     for i in range(1, 5):
#         roi = np.where(mask_arr == i)
#         df = pd.read_csv(os.path.join(r"C:\Users\jwu191\Desktop", "phantom "+str(i)+".csv"))
#         data = np.array(df)
#         for j in range(roi[0].shape[0]):
#             img[roi[0][j]][roi[1][j]] = data[j][13]
#     img = np.flip(img, 0) # axis=0 !
#     print(np.min(img), np.max(img))
#     img[img > 5] = 5
#     img[img < -5] = -5
#     img[0][0] = 5
#     img[0][1] = -5
#     img = np.interp(img, (img.min(), img.max()), (0, 255))
#     img[0][0] = 0
#     img[0][1] = 0
#     cv2.imwrite(r"C:\Users\jwu191\Desktop\Lorentz.png", img)  
#     img = cv2.imread(r"C:\Users\jwu191\Desktop\Lorentz.png", 1)
#     lookup_table = read_lookup_table(r"C:\Users\jwu191\Desktop\my_pipeline\idl_rainbow.lut")
#     img = gray_to_idl(img, lookup_table)
#     cv2.imwrite(r"C:\Users\jwu191\Desktop\Lorentz.png", img) 


# roi = get_phantom_mask_all(nth_slice)
# img_path = os.path.join(processed_cases_dir, patient_name+"_all", patient_name,
#                         "4_coreg2apt", patient_name, patient_name+"_Flair.nii")
# img = sitk.ReadImage(img_path, sitk.sitkInt32)
# arr = sitk.GetArrayFromImage(img)[nth_slice-1]

# print(arr.shape)
# print(np.min(arr), np.max(arr))
# # val = np.min(arr)
# # arr[roi] = val
# arr = np.flip(arr, 0) # axis=0 !
# arr = np.interp(arr, (arr.min(), arr.max()), (0, 255))
# cv2.imwrite(r"C:\Users\jwu191\Desktop\Flair.png", arr) 

# mapping_dir = r"C:\Users\jwu191\Desktop\mapping_output"
# output_dir = r"C:\Users\jwu191\Desktop\output"
# patient_name = '20220815_SKC'
# nthslice = 9
# B1_index = 0 # In ,ost cases, 0 for 1.5uT, 1 for 2uT
# emr_offset = [np.inf, 80, 60, 40, 30, 20, 12, 8, 4, -4, 3.5, -3.5,
#                   3.5, -3.5, 3, -3, 2.5, -2.5, 2, -2, 2, -2, 1.5, -1.5]    
# scanned_offset = [np.inf, 0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1, -1, 1.25, -1.25, 
#                   1.5, -1.5, 1.75, -1.75, 2, -2, 2.25, -2.25, 2.5, -2.5, 2.75, -2.75, 
#                   3, -3, 3.25, -3.25, 3.5, -3.5, 3.75, -3.75, 4, -4, 4.25, -4.25, 
#                   4.5, -4.5, 4.75, -4.75, 5, -5, 8, -8, 12, -12, 20, -20, 30, -30,
#                   40, -40, 60, -60, 80, -80]
# # selected_offset = [8, -8, 12, -12, 20, -20, 30, -30, 40, -40, 60, -60, 80, -80]
# middle_offset = [0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1, -1, 1.25, -1.25, 
#                   1.5, -1.5, 1.75, -1.75, 2, -2, 2.25, -2.25, 2.5, -2.5, 2.75, -2.75, 
#                   3, -3, 3.25, -3.25, 3.5, -3.5, 3.75, -3.75, 4, -4, 4.25, -4.25, 
#                   4.5, -4.5, 4.75, -4.75, 5, -5]
# all_offset = [0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1, -1, 1.25, -1.25, 
#                   1.5, -1.5, 1.75, -1.75, 2, -2, 2.25, -2.25, 2.5, -2.5, 2.75, -2.75, 
#                   3, -3, 3.25, -3.25, 3.5, -3.5, 3.75, -3.75, 4, -4, 4.25, -4.25, 
#                   4.5, -4.5, 4.75, -4.75, 5, -5, 8, -8, 12, -12, 20, -20, 30, -30,
#                   40, -40, 60, -60, 80, -80]

# t1map_array, t2map_array, zspectra_all, zspectra_name = ReadData_Zspec(os.path.join(mapping_dir, patient_name), 
#                                                                        nthslice)
# print('T1 map:', t1map_array.shape)
# print('T2 map:', t2map_array.shape)
# print('Zspec all:', zspectra_all.shape)
# print('Zspec name:', zspectra_name)      
        
# roi, size = get_normal_mask(output_dir, patient_name, nthslice)
# data = []
# offset = None
# for i in range(size):
#     zspectra = zspectra_all[0, :, roi[0][i], roi[1][i]] # 0: 1.5uT
#     Zspec, offset = Zspec_preprocess_onepixel(zspectra, scanned_offset, all_offset)
#     plot_Zspec(offset, Zspec/100, patient_name+' '+str(i)+' normal')
#     item = [roi[0][i], roi[1][i]] + list(Zspec)
#     data.append(item)
# column_names = ['x', 'y']
# for i in range(len(offset)):
#     column_names.append(str(offset[i]))
# data = np.array(data)
# df = pd.DataFrame(data=data, columns=column_names)
# save_path = os.path.join(r"C:\Users\jwu191\Desktop", 
#                           patient_name+'_normal_Zspec.csv')
# df.to_csv(save_path)        


# Zspec = get_avg_Zspec(r"C:\Users\jwu191\Desktop\EMR fitting\Zspec_1.5uT_56offsets\20220721_DW_normal_Zspec.csv")
# print(Zspec)
# plot_Zspec(offset, Zspec/100, "20220720_JG_normal_AVG")








        
        