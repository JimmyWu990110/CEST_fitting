import os

import numpy as np
from scipy.optimize import curve_fit


class Lorentz_fitting:
    def __init__(self, initialization): 
        self.initialization = initialization
    
    def lorentz(self, x, amp, pos, width):
        factor = ((x - pos) / width) * ((x - pos) / width)
        return amp / (1 + 4*factor)
    
    def func_2_pool(self, x, amp_mt, pos_mt, width_mt,
                    amp_ds, pos_ds, width_ds):
        mt = self.lorentz(x, amp_mt, pos_mt, width_mt)
        ds = self.lorentz(x, amp_ds, pos_ds, width_ds)
        return 1 - (mt + ds)
    
    def fit_2_pool(self, offset, Zspec):
        popt, pcov = curve_fit(self.func_2_pool, xdata=offset, ydata=Zspec, 
                               p0=self.initialization.x0,
                               bounds=(self.initialization.lb, self.initialization.ub), 
                               method="trf", maxfev=5000)
        for i in range(2):
            popt[3*i + 1] -= self.initialization.B0_shift
        return popt
    
    def generate_Zpsec(self, model_type, offset, paras):
        y = []
        if model_type == "2_pool":
            for x in offset: 
                y.append(self.func_2_pool(x, *paras))
        return np.array(y)

    
    def func_5_pool(self, x, amp_noe, pos_noe, width_noe,
                   amp_mt, pos_mt, width_mt,
                   amp_ds, pos_ds, width_ds, 
                   amp_cest, pos_cest, width_cest,
                   amp_apt, pos_apt, width_apt):
        noe = self.lorentz(x, amp_noe, pos_noe, width_noe)
        mt = self.lorentz(x, amp_mt, pos_mt, width_mt)
        ds = self.lorentz(x, amp_ds, pos_ds, width_ds)
        cest = self.lorentz(x, amp_cest, pos_cest, width_cest)
        apt = self.lorentz(x, amp_apt, pos_apt, width_apt)
        return 100 - (noe + mt + ds + cest + apt)
    
    def func_fixed_5_pool(self, x, amp_noe, width_noe, amp_mt, width_mt, 
                         amp_ds, width_ds, amp_cest, width_cest, amp_apt, width_apt):
        noe = self.lorentz(x, amp_noe, -3.5, width_noe)
        mt = self.lorentz(x, amp_mt, -2.34, width_mt)
        ds = self.lorentz(x, amp_ds, 0, width_ds)
        cest = self.lorentz(x, amp_cest, 2, width_cest)
        apt = self.lorentz(x, amp_apt, 3.5, width_apt)
        return 100 - (noe + mt + ds + cest + apt)
    




    

  


# def plot_5pool(offset, noe, mt, ds, cest, apt, zspec, title):
#     plt.ylim((0, 100))
#     plt.scatter(offset, noe, label='noe', color='green')
#     plt.scatter(offset, mt, label='mt', color='purple')
#     plt.scatter(offset, ds, label='ds', color='blue')
#     plt.scatter(offset, cest, label='cest', color='yellow')
#     plt.scatter(offset, apt, label='apt', color='red')
#     plt.scatter(offset, zspec, label='zspec', color='black')
#     plt.gca().invert_xaxis()  # invert x-axis
#     plt.legend()
#     plt.title(title)
#     plt.show()    
    
# def plot_fixed_5pool(offset, paras, title):
#     noe_paras = paras[0:2]
#     mt_paras = paras[2:4]
#     ds_paras = paras[4:6]
#     cest_paras = paras[6:8]
#     apt_paras = paras[8:10]
#     noe = []
#     mt = []
#     ds = []
#     cest = []
#     apt = []
#     zspec = []
#     for x in offset:
#         noe.append(lorentz(x, noe_paras[0], -3.5, noe_paras[1]))
#         mt.append(lorentz(x, mt_paras[0], -1.5, mt_paras[1]))
#         ds.append(lorentz(x, ds_paras[0], 0, ds_paras[1]))
#         cest.append(lorentz(x, cest_paras[0], 2, cest_paras[1]))
#         apt.append(lorentz(x, apt_paras[0], 3.5, apt_paras[1]))
#         zspec.append(100 - (noe[-1]+mt[-1]+ds[-1]+cest[-1]+apt[-1]))
    
#     plt.scatter(offset, noe, label='noe', color='green')
#     plt.scatter(offset, mt, label='mt', color='purple')
#     plt.scatter(offset, ds, label='ds', color='blue')
#     plt.scatter(offset, cest, label='cest', color='yellow')
#     plt.scatter(offset, apt, label='apt', color='red')
#     plt.scatter(offset, zspec, label='zspec', color='black')
#     plt.gca().invert_xaxis()  # invert x-axis
#     plt.legend()
#     plt.title(title)
#     plt.show()

# def fitOnePixel(free, offset, zarray, method, initials, num_iter, bounds, title, b0_shift):
#     popt = None
#     popt_uncorrected = None
#     if free:
#         popt, pcov = curve_fit(func_5pool, xdata=offset, ydata=zarray, method=method,
#                            p0=initials, maxfev=num_iter, bounds=bounds)
#         popt_uncorrected = popt.copy()
#         popt[1] -= b0_shift; popt[4] -= b0_shift; popt[7] -= b0_shift;
#         popt[10] -= b0_shift; popt[13] -= b0_shift;
#     else:
#         popt, pcov = curve_fit(func_fixed_5pool, xdata=offset, ydata=zarray, method=method,
#                            p0=initials, maxfev=num_iter, bounds=bounds)
#         paras = np.zeros(15)
#         paras[0] = popt[0]; paras[1] = -3.5 - b0_shift; paras[2] = popt[1];
#         paras[3] = popt[2]; paras[4] = -2.34 - b0_shift; paras[5] = popt[3];
#         paras[6] = popt[4]; paras[7] = 0 - b0_shift; paras[8] = popt[5];
#         paras[9] = popt[6]; paras[10] = 2 - b0_shift; paras[11] = popt[7];
#         paras[12] = popt[8]; paras[13] = 3.5 - b0_shift; paras[14] = popt[9];
#         popt = paras

#     return popt







