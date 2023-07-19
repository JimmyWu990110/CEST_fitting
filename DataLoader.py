import os
import numpy as np
import pandas as pd
import SimpleITK as sitk

class ZspecOffset56:
    full_56 = np.array([np.inf, 0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1, -1, 1.25, -1.25, 
                        1.5, -1.5, 1.75, -1.75, 2, -2, 2.25, -2.25, 2.5, -2.5, 2.75, -2.75, 
                        3, -3, 3.25, -3.25, 3.5, -3.5, 3.75, -3.75, 4, -4, 4.25, -4.25, 
                        4.5, -4.5, 4.75, -4.75, 5, -5, 8, -8, 12, -12, 20, -20, 30, -30,
                        40, -40, 60, -60, 80, -80])
    # remove inf
    full_55 = np.array([0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1, -1, 1.25, -1.25, 1.5, -1.5, 
                        1.75, -1.75, 2, -2, 2.25, -2.25, 2.5, -2.5, 2.75, -2.75, 3, -3, 3.25, -3.25, 
                        3.5, -3.5, 3.75, -3.75, 4, -4, 4.25, -4.25, 4.5, -4.5, 4.75, -4.75, 5, -5, 
                        8, -8, 12, -12, 20, -20, 30, -30, 40, -40, 60, -60, 80, -80])
    middle_45 = np.array([0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1, -1, 1.25, -1.25, 1.5, -1.5, 
                          1.75, -1.75, 2, -2, 2.25, -2.25, 2.5, -2.5, 2.75, -2.75, 3, -3, 3.25, -3.25, 
                          3.5, -3.5, 3.75, -3.75, 4, -4, 4.25, -4.25, 4.5, -4.5, 4.75, -4.75, 5, -5,
                          8, -8, 12, -12])
    middle_41 = np.array([0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1, -1, 1.25, -1.25, 1.5, -1.5, 
                          1.75, -1.75, 2, -2, 2.25, -2.25, 2.5, -2.5, 2.75, -2.75, 3, -3, 3.25, -3.25, 
                          3.5, -3.5, 3.75, -3.75, 4, -4, 4.25, -4.25, 4.5, -4.5, 4.75, -4.75, 5, -5])
    # remove 0
    middle_40 = np.array([0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1, -1, 1.25, -1.25, 1.5, -1.5, 
                          1.75, -1.75, 2, -2, 2.25, -2.25, 2.5, -2.5, 2.75, -2.75, 3, -3, 3.25, -3.25, 
                          3.5, -3.5, 3.75, -3.75, 4, -4, 4.25, -4.25, 4.5, -4.5, 4.75, -4.75, 5, -5])
    wide_16 = np.array([5, -5, 8, -8, 12, -12, 20, -20, 30, -30, 40, -40, 60, -60, 80, -80])
    wide_14 = np.array([8, -8, 12, -12, 20, -20, 30, -30, 40, -40, 60, -60, 80, -80])
    wide_12 = np.array([12, -12, 20, -20, 30, -30, 40, -40, 60, -60, 80, -80])
    wide_10 = np.array([20, -20, 30, -30, 40, -40, 60, -60, 80, -80])
    wide_downfield_8 = np.array([80, 60, 40, 30, 20, 12, 8, 5])
    wide_downfield_7 = np.array([80, 60, 40, 30, 20, 12, 8])
    wide_downfield_6 = np.array([80, 60, 40, 30, 20, 12])
    wide_downfield_5 = np.array([80, 60, 40, 30, 20])
    combine_7_1 = np.array([0.5, 80, 60, 40, 30, 20, 12, 8])
    
    def __init__(self):
        pass


class DataLoader:
    # From 2022/03/24, EMR offset 10ppm changed to 12ppm
    EMR_24_offset_ppm = np.array([np.inf, 80, 60, 40, 30, 20, 12, 8, 4, -4, 3.5, -3.5, 3.5, -3.5, 
                  3, -3, 2.5, -2.5, 2, -2, 2, -2, 1.5, -1.5])
    Zspec_offset = ZspecOffset56()
    def __init__(self, patient_name, B1=1.5):
        self.patient_name = patient_name
        self.mask_dir = r"C:\Users\jwu191\Desktop\EMR fitting\Zspec_mask"
        self.processed_cases_dir = r"C:\Users\jwu191\Desktop\processed_cases"
        self.data_dir = os.path.join(r"C:\Users\jwu191\Desktop\processed_cases",
                                     patient_name+"_all", patient_name+"_mapping")
        self.seq_dict = None
        self.nth_slice = -1
        self.Zspec_img = None
        self.T1_map = None
        self.T2_map = None

        self.get_seq_dict()
        self.get_nth_slice_by_Zspec_mask()
        # self.get_ROIs_by_Zspec_mask()
        self.read_Zspec(B1)
        # 2D, needs nth_slcie
        self.get_T1_map()
        self.get_T2_map()
        
    def get_seq_dict(self):
        df = pd.read_excel(os.path.join(self.processed_cases_dir, self.patient_name+"_all", 
                                        self.patient_name, "all_scan_info.xlsx"))
        seq_ids = list(df["Sequence_id"])
        protocols = list(df["Protocol"]) # same length with seq_id list
        EMR_1p5uT_id = -1
        WASSR_EMR_id = -1
        EMR_2uT_id = -1
        Zspec_1p5uT_id = -1
        WASSR_Zspec_id = -1
        Zspec_2uT_id = -1
        APTw_id = -1
        APTw_CS4_id = -1
        for i in range(len(protocols)):
            protocol = protocols[i].lower()
            current_id = int(seq_ids[i].split("_")[-2])
            if "emrns1p5" in protocol and "zspec" not in protocol:
                EMR_1p5uT_id = current_id
                next_protocol = protocols[i+1].lower()
                if "wassr" in next_protocol:
                    WASSR_EMR_id = current_id + 1
            if "emrns2" in protocol and "zspec" not in protocol:
                EMR_2uT_id = current_id
            if "zspecemr_single" in protocol and "1p5ut" in protocol:
                Zspec_1p5uT_id = current_id  
                previous_protocol = protocols[i-1].lower()
                if "wassr" in previous_protocol:
                    WASSR_Zspec_id = current_id - 1
            if "zspecemr_single" in protocol and "2ut" in protocol:
                Zspec_2uT_id = current_id
            if "aptw_2ut" in protocol and "cs4" not in protocol:
                APTw_id = current_id
            if "aptw_2ut" in protocol and "cs4" in protocol:
                APTw_CS4_id = current_id  
        self.seq_dict = {"EMR_1p5uT_id":EMR_1p5uT_id, "WASSR_EMR_id":WASSR_EMR_id, 
                         "EMR_2uT_id":EMR_2uT_id,
                         "Zspec_1p5uT_id":Zspec_1p5uT_id, "WASSR_Zspec_id":WASSR_Zspec_id, 
                         "Zspec_2uT_id":Zspec_2uT_id,
                         "APTw_id":APTw_id, "APTw_CS4_id":APTw_CS4_id}
        print("sequences:", self.seq_dict)
              
    def get_nth_slice_by_Zspec_mask(self):
        # Zspec mask: 1 slice simplified mask for Zspec
        mask_path = os.path.join(self.mask_dir, self.patient_name+".nii.gz")
        mask = sitk.ReadImage(mask_path, sitk.sitkInt32) # mask: Int
        mask_arr = sitk.GetArrayFromImage(mask)
        print("mask shape:", mask_arr.shape)
        for i in range(mask_arr.shape[0]):
            # 1 for tumor and 2 for normal
            if np.sum(mask_arr[i] == 1) > 0 and np.sum(mask_arr[i] == 2) > 0:
                print("nth_slice:", i + 1)
                self.nth_slice = i + 1
                return
        raise Exception("Invalid Zspec mask!")

    def get_ROIs_by_Zspec_mask(self):
        self.get_nth_slice_by_Zspec_mask()
        mask_path = os.path.join(self.mask_dir, self.patient_name+".nii.gz")
        mask = sitk.ReadImage(mask_path, sitk.sitkInt32)
        mask_arr = sitk.GetArrayFromImage(mask)[self.nth_slice-1]
        tumor_roi = np.where(mask_arr == 1)
        normal_roi = np.where(mask_arr == 2)
        print("tumor size:", tumor_roi[0].shape[0])
        print("normal size:", normal_roi[0].shape[0])
        return tumor_roi, normal_roi
    
    def get_ROIs_by_phantom_mask(self, label):
        mask_path = os.path.join(self.processed_cases_dir, self.patient_name+"_all",
                                 self.patient_name, "4_coreg2apt", self.patient_name, 
                                 self.patient_name+"_mask.nii.gz")
        mask = sitk.ReadImage(mask_path, sitk.sitkInt32)
        mask_array = sitk.GetArrayFromImage(mask)[self.nth_slice-1]
        roi = None
        if label == 9999:
            roi = np.where(mask_array != 0)
        else:
            roi = np.where(mask_array == label)
        print("phantom size:", roi[0].shape[0])
        return roi
        
    def read_Zspec(self, B1):
        print("******** read Zspec ********")
        EMR_path = os.path.join(self.data_dir, "EMR-nifti")
        B1_str = None
        if B1 == 1.5:
            B1_str = "1p5uT"
        elif B1 == 2:
            B1_str = "2uT"
        else:
            raise Exception("Invalid B1, only supports 1.5 and 2.")
        for f in os.listdir(EMR_path):
            if "ZspecEMR" in f and B1_str in f and f.endswith('.nii'):
                print(f, "readed")
                Zspec = sitk.ReadImage(os.path.join(EMR_path, f))
                Zspec_arr = sitk.GetArrayFromImage(Zspec)
                # [56, 1, 256, 256] -> [56, 256, 256]
                self.Zspec_img = np.squeeze(Zspec_arr)
                print("Zspec img shape:", self.Zspec_img.shape)
                return
        raise Exception("Zspec file not found!")
        
    def get_WASSR_Zspec(self):
        print("******** read WASSR for Zspec ********")
        seq_id = self.seq_dict["WASSR_Zspec_id"]
        base_dir = os.path.join(self.data_dir, "WASSR-nifti")
        files = os.listdir(base_dir)
        for f in files:
            if f.endswith("_"+str(seq_id)+".nii"):
                # return os.path.join(base_dir, f)
                img = sitk.ReadImage(os.path.join(base_dir, f))
                arr = sitk.GetArrayFromImage(img)
                print("read", f, "shape", arr.shape)
                return arr
        raise Exception("WASSR_Zspec file not found!")
    
    @staticmethod
    def Zspec_preprocess_onepixel(Zspec, scanned_offset, selected_offset):
        """
        sort Zspec with selected offsets from positive to negative
        then normalize it by M0
        """
        scanned_offset = np.array(scanned_offset)
        selected_offset = np.array(selected_offset)
        sort_index = np.argsort(selected_offset)[::-1]
        selected_offset = selected_offset[sort_index]
        processed_Zspec = []
        M0 = Zspec[np.where(scanned_offset==np.inf)][0]
        for x in selected_offset:
            Mx = Zspec[np.where(scanned_offset == x)]
            processed_Zspec.append(np.mean(Mx) / M0)
        return np.array(processed_Zspec), selected_offset

    def get_tumor_avg_Zspec(self, offset):
        Zspec = []
        for x in offset:
            data = self.Zspec_img[np.where(self.Zspec_56_offset_ppm==x)][0] # 256*256
            Zspec.append(np.mean(data[self.tumor_roi]))
        M0 = self.Zspec_img[np.where(self.Zspec_56_offset_ppm==np.inf)][0] # 256*256
        Zspec /= np.mean(M0[self.tumor_roi])
        return np.array(Zspec)
    
    def get_normal_avg_Zspec(self, offset):
        Zspec = []
        for x in offset:
            data = self.Zspec_img[np.where(self.Zspec_56_offset_ppm==x)][0] # 256*256
            Zspec.append(np.mean(data[self.normal_roi]))
        M0 = self.Zspec_img[np.where(self.Zspec_56_offset_ppm==np.inf)][0] # 256*256
        Zspec /= np.mean(M0[self.tumor_roi])
        return np.array(Zspec)
    
    def get_T1_map(self):
        img = sitk.ReadImage(os.path.join(self.data_dir, "t1_map.nii"))
        self.T1_map = sitk.GetArrayFromImage(img)[self.nth_slice]
        
    def get_T2_map(self):
        img = sitk.ReadImage(os.path.join(self.data_dir, "t2_map.nii"))
        self.T2_map = sitk.GetArrayFromImage(img)[self.nth_slice]


def get_phantom_mask(patient_name, nth_slice, phantom_label):
    mask_path = os.path.join(r"C:\Users\jwu191\Desktop\processed_cases", patient_name+"_all",
                 patient_name, "4_coreg2apt", patient_name, patient_name+"_mask.nii.gz")
    # mask_path = r"C:\Users\jwu191\Desktop\processed_cases\20230309_phantom_all\20230309_phantom\4_coreg2apt\20230309_phantom\20230309_phantom_mask.nii.gz"
    mask = sitk.ReadImage(mask_path, sitk.sitkInt32)
    mask_array = sitk.GetArrayFromImage(mask)[nth_slice-1]
    return np.where(mask_array == phantom_label)

def get_phantom_mask_all(nth_slice):
    mask_path = r"C:\Users\jwu191\Desktop\processed_cases\20230309_phantom_all\20230309_phantom\4_coreg2apt\20230309_phantom\20230309_phantom_mask.nii.gz"
    mask = sitk.ReadImage(mask_path, sitk.sitkInt32)
    mask_array = sitk.GetArrayFromImage(mask)[nth_slice-1]
    return np.where(mask_array == 0)

def read_Zspec_by_id(base_dir, seq_id):
    print("******** read Zspec ********")
    EMR_path = os.path.join(base_dir, "EMR-nifti")
    for f in os.listdir(EMR_path):
        if "ZspecEMR" in f and f.endswith(str(seq_id)+".nii"):
            print("read", f)
            Zspec_img = sitk.ReadImage(os.path.join(EMR_path, f))
            Zspec_array = sitk.GetArrayFromImage(Zspec_img)
            return np.squeeze(Zspec_array)
    raise Exception("Zspec file not found!")

def offset_preprocess(offset):
    offset = np.array(offset)
    sorted_index = np.argsort(offset)[::-1]
    return offset[sorted_index]


        




    


  