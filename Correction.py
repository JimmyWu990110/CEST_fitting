import cv2
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

class B0_Correction:
    def __init__(self, WASSR_img):
        self.freq = np.array([np.inf, 0, 14, -14, 28, -28, 42,-42, 56, -56, 70, -70, 84, -84,
                              98, -98, 112, -112, 126, -126, 140, -140, 154, -154, 168, -168])
        self.WASSR_img = WASSR_img
        
    def polyfit_onepixel(self, freq, y):
        """
        given a freq and y, use poly fit to find the index of its lowest point
        (x upsampled to stride=1), the index is the B0_shift
        """
        sort_index = np.argsort(freq)
        freq_sorted = np.sort(freq)
        y_sorted = y[sort_index]
        x_upsampled = np.arange(-168, 169, 1) #[-168, -167, ... 168]
        paras = np.polyfit(freq_sorted, y_sorted, deg=8)
        p = np.poly1d(paras)
        index = np.argmin(p(x_upsampled))
        # if abs(x_upsampled[index]) > -1:
        #     plt.scatter(freq_sorted, y_sorted)
        #     plt.plot(x_upsampled, p(x_upsampled))
        #     plt.show()
        return x_upsampled[index]
    
    def cal_B0_shift_onepixel(self, nth_slice, x, y):
        """
        given the nth_slice and (x, y) coordinates (in APT img with shape 256*256),
        calculate the B0 shift at corresponding location using the WASSR img (shape 128*128)
        """
        WASSR = self.WASSR_img[:, nth_slice-1, :, :] # one slice, 26*128*128
        M0 = WASSR[0] # one slice, 128*128
        # remove np.inf and M0
        used_freq = self.freq[1:] # len:25
        WASSR = WASSR[1:] # 25*128*128
        # 256 -> 128: /2, normalized by M0
        normalzied_WASSR_onepixel = WASSR[:, int(x/2), int(y/2)] / M0[int(x/2), int(y/2)]
        B0_shift = self.polyfit_onepixel(used_freq, normalzied_WASSR_onepixel)
        return B0_shift

    
    def polyFitZspec(self, zspec, b0_shift, offset):
        offset = offset * 128
        paras = np.polyfit(offset, zspec, deg=12)
        p = np.poly1d(paras)
        z_fitted = []
        for x in offset:
            z_fitted.append(p(x + b0_shift))
        plt.ylim((0, 100))
        plt.scatter(offset, zspec, label='src', color='blue')
        plt.scatter(offset, z_fitted, label='fitted', color='red')
        plt.legend()
        plt.gca().invert_xaxis()  # invert x-axis
        plt.show()
        return z_fitted
    
    def get_B0_map(self, path, n_slice):
        scanned_offset = [np.inf, 0, 14, -14, 28, -28, 42,-42, 56, -56, 70, -70, 84, -84,
        98, -98, 112, -112, 126, -126, 140, -140, 154, -154, 168, -168]; # len:26
        wassr_img = sitk.ReadImage(path)
        wassr_array_raw = sitk.GetArrayFromImage(wassr_img)[:, n_slice-1, :, :] # 26*128*128
        M0 = wassr_array_raw[0] # 128*128
        desired_offset = scanned_offset[1:] # len:25
        wassr_array = wassr_array_raw[1:]
        
        # M0 = cv2.blur(M0, (3,3))
        # mask_these = np.where(M0 < 5000)
        # M0[mask_these] = 0
        # plt.imshow(M0, cmap='gray')
        # plt.show()
        B0_map = np.zeros((M0.shape[0], M0.shape[1]))
        for i in range(M0.shape[0]):
            for j in range(M0.shape[1]):
                if M0[i][j] > 0:
                    B0_map[i][j] = polyFitOnePixel(img=wassr_array, x=i, y=j, offset=desired_offset)
                
        # plt.imshow(B0_map, cmap='gray')
        # plt.show()
        # plt.hist(B0_map)
        # plt.show()
        return B0_map
    
    def B0CorrOnePixel(self, path, x, y, n_slice, zarray, offset):
        scanned_offset = [np.inf, 0, 14, -14, 28, -28, 42,-42, 56, -56, 70, -70, 84, -84,
        98, -98, 112, -112, 126, -126, 140, -140, 154, -154, 168, -168]; # len:26
        wassr_img = sitk.ReadImage(path)
        wassr_array_raw = sitk.GetArrayFromImage(wassr_img)[:, n_slice-1, :, :] # 26*128*128
        m0 = wassr_array_raw[0] 
        print(m0.shape)
        desired_offset = scanned_offset[1:] # len:25
        wassr_array = wassr_array_raw[1:] # len:25
        print(wassr_array.shape)
        b0 =  polyFitOnePixel(img=wassr_array, x=int(x/2), y=int(y/2), offset=desired_offset)
        print('b0:', b0)
        z_fitted = polyFitZspec(zspec=zarray[:, x, y], b0_shift=b0, offset=offset)
        return z_fitted
    
    def find_points(self, x, acq_offset_hz):
        x1 = 9999
        x2 = 9999
        for i in range(acq_offset_hz.shape[0]):
            if acq_offset_hz[i] >= x:
                x1 = acq_offset_hz[i]
        for i in range(acq_offset_hz.shape[0]):
            if acq_offset_hz[i] <= x:
                x2 = acq_offset_hz[i]
                break
        return x1, x2

    def interpolate_B0_shift(self, offset_hz, Zspec, b0_shift):
        acq_offset_hz = offset_hz - b0_shift
        # print("offset_hz:", offset_hz)
        # print("(actually acuqired) offset_hz:", acq_offset_hz)
        # print("Zspec:", Zspec)
        # print("b0_shift:", b0_shift)
        Zspec_corrected = np.ones(Zspec.shape[0]) + 1
        for i in range(offset_hz.shape[0]):
            x = int(offset_hz[i])
            x1, x2 = find_points(x, acq_offset_hz)
            # print("x, x1, x2:", x, x1, x2)
            if x1 != 9999 and x2 != 9999:
                y1 = Zspec[np.where(acq_offset_hz == int(x1))[0][0]]
                y2 = Zspec[np.where(acq_offset_hz == int(x2))[0][0]]
                if x1 == x2:
                    Zspec_corrected[i] = y1
                else:
                    y = y1 + ((y2-y1)*(x-x1)) / (x2-x1)
                    Zspec_corrected[i] = y
            elif x1 == 9999:
                x3 = acq_offset_hz[np.where(acq_offset_hz == int(x2))[0][0] + 1]
                # print("x, x2, x3:", x, x2, x3)
                y2 = Zspec[np.where(acq_offset_hz == int(x2))[0][0]]
                y3 = Zspec[np.where(acq_offset_hz == int(x3))[0][0]]
                # print("y2, y3:", y2, y3)
                y = y2 + ((y2-y3)*(x-x2)) / (x2-x3)
                Zspec_corrected[i] = y
            elif x2 == 9999:
                x0 = acq_offset_hz[np.where(acq_offset_hz == int(x1))[0][0] - 1]
                y1 = Zspec[np.where(acq_offset_hz == int(x1))[0][0]]
                y0 = Zspec[np.where(acq_offset_hz == int(x0))[0][0]]
                y = y1 + ((y1-y0)*(x-x1)) / (x1-x0)
                Zspec_corrected[i] = y
        # flip to fill the empty offsets, <= 5 points (160hz)
        # for i in range(5):
        #     if Zspec_corrected[i] == 2:
        #         Zspec_corrected[i] = Zspec_corrected[-i-1]
        #     if Zspec_corrected[-i-1] == 2:
        #         Zspec_corrected[-i-1] = Zspec_corrected[i]
        
        # print("Zspec_corrected:", Zspec_corrected)    
        # plt.scatter(offset_hz, Zspec, label="Zspec", color="black")
        # plt.scatter(offset_hz, Zspec_corrected, label="Zspec_corrected", color="red")
        # plt.gca().invert_xaxis()  # invert x-axis
        # plt.legend()
        # plt.show()
        return Zspec_corrected





    















