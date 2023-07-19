import math

import numpy as np
from scipy.optimize import curve_fit
import scipy.integrate as integrate   

    
class EMR_fitting:
    def __init__(self, T1w_obs, T2w_obs, B1=1.5):
        self.model_type = None
        self.B1 = B1
        self.T1w_obs = T1w_obs
        self.T2w_obs = T2w_obs
        self.T1m = 1
        self.x0 = None
        self.lb = None
        self.ub = None
        self.lineshape = "SL"
        self.R = 20 # for fixed R
        self.fitted_paras = None
        
    def set_model_type(self, model_type):
        if model_type == 0 or model_type == 1 or model_type == 2:
            self.model_type = model_type
        else:
            raise Exception("Invalid model type! Please use 0, 1 or 2")
   
    def set_x0(self, x0):
        self.x0 = x0
        
    def set_lb(self, lb):
        self.lb = lb
        
    def set_ub(self, ub):
        self.ub = ub
   
    def set_lineshape(self, lineshape):
        if lineshape not in ["G", "L", "SL"]:
            raise Exception("Invalid lineshape! Please use G, L or SL")
        self.lineshape = lineshape
    
    def _func_gaussian(self, x, T2m):
        term = 2 * math.pi * x * T2m
        exp_term = math.exp(-term*term*0.5)
        return (T2m / math.sqrt(2*math.pi)) * exp_term
                
    def _func_lorentz(self, x, T2m):
        term = 2 * math.pi * x * T2m
        return (T2m/math.pi) * (1 / (1 + term*term))

    def _func_super_lorentz(self, t, x, T2m):
        # t is the independent variable for integration, (x, T2m) are parameters
        cos_denominator = abs(3*math.cos(t)*math.cos(t) - 1)
        term = math.sin(t) * math.sqrt(2/math.pi) * (T2m/cos_denominator)
        T2m_numerator = 2 * math.pi * x * T2m 
        power = (-2) * (T2m_numerator/cos_denominator) * (T2m_numerator/cos_denominator)
        return term * math.exp(power) 
    
    def cal_Rrf(self, freq, T2, lineshape):
        result = []
        for x in freq:
            if lineshape == 'G':
                val = self._func_gaussian(x, T2)
                result.append(val)
            elif lineshape == 'L':
                val = self._func_lorentz(x, T2)
                result.append(val)
            elif lineshape == 'SL':
                val, err = integrate.quad(self._func_super_lorentz, 0, math.pi/2, args=(x, T2))
                result.append(val)  
        w1 = 267.522 * self.B1
        return np.array(result) * w1 * w1 * math.pi
    
    def MT_model(self, freq, A, B, C, D):
        # [R, R*M0m*T1w, T1w/T2w, T2m]
        """
        given frequencies and these 4 paras, return Zspec(Mz/M0)
        """
        w1 = 267.522 * self.B1
        Rrfm_line = self.cal_Rrf(freq, D, self.lineshape)
        Zspec = []
        for i in range(freq.shape[0]):
            x = freq[i]
            Rrfm = Rrfm_line[i]
            if x == 0: # deal with singular point
                Zspec.append(0)
                continue    
            # following the formula
            tmp = w1 / (2*math.pi*x)
            numerator = B + (Rrfm + A + 1)
            denominator_1 = B * (Rrfm+1)
            denominator_2 = (1 + tmp*tmp*C) * (Rrfm + A + 1)
            Zspec.append(numerator / (denominator_1+denominator_2))
        return np.array(Zspec)
    
    def MT_model_1(self, freq, A, B, C, D):
        # [R, M0m*T1w, T1w/T2w, T2m]
        """
        given frequencies and these 4 paras, return Zspec(Mz/M0)
        """
        w1 = 267.522 * self.B1
        absorption_line = self.cal_Rrf(freq, D, self.lineshape)
        Rrfm_line = self.cal_Rrf(freq, D, self.lineshape)
        Zspec = []
        for i in range(freq.shape[0]):
            x = freq[i]
            Rrfm = Rrfm_line[i]
            if x == 0: # deal with singular point
                Zspec.append(0)
                continue    
            # following the formula
            tmp = w1 / (2*math.pi*x)
            numerator = A*B + (Rrfm + A + 1)
            denominator_1 = A*B * (Rrfm + 1)
            denominator_2 = (1 + tmp*tmp*C) * (Rrfm + A + 1)
            Zspec.append(numerator / (denominator_1+denominator_2))
        return np.array(Zspec)    
    
    def MT_model_2(self, freq, B, C, D):
        # [M0m*T1w, T1w/T2w, T2m]
        """
        given frequencies and these 4 paras, return Zspec(Mz/M0)
        """
        A = self.R # fixed R
        w1 = 267.522 * self.B1
        absorption_line = self.cal_Rrf(freq, D, self.lineshape)
        Rrfm_line = self.cal_Rrf(freq, D, self.lineshape)
        Zspec = []
        for i in range(freq.shape[0]):
            x = freq[i]
            Rrfm = Rrfm_line[i]
            if x == 0: # deal with singular point
                Zspec.append(0)
                continue    
            # following the formula
            tmp = w1 / (2*math.pi*x)
            numerator = A*B + (Rrfm + A + 1)
            denominator_1 = A*B * (Rrfm + 1)
            denominator_2 = (1 + tmp*tmp*C) * (Rrfm + A + 1)
            Zspec.append(numerator / (denominator_1+denominator_2))
        return np.array(Zspec) 
    
    # def fit(self, freq, Zspec, constrained):
    #     if math.isnan(self.T1w_obs/self.T2w_obs) or self.T1w_obs/self.T2w_obs == 0 or math.isinf(self.T1w_obs/self.T2w_obs):
    #         raise Exception("Invalid T1/T2!")
    #     else: 
    #         popt, pcov = None, None
    #         if constrained:
    #             popt, pcov = curve_fit(self.MT_model, xdata=freq, ydata=Zspec, 
    #                                    p0=self.x0, bounds=(self.lb, self.ub), 
    #                                    method='trf', maxfev=5000)
    #         else:
    #             popt, pcov = curve_fit(self.MT_model, xdata=freq, ydata=Zspec, 
    #                                    p0=self.x0, 
    #                                    method='lm', maxfev=5000)
    #         self.fitted_paras = popt
    #         y_estimated = self.MT_model(freq, *popt)
    #         return popt, y_estimated
         
    def fit(self, freq, Zspec, constrained):
        if math.isnan(self.T1w_obs/self.T2w_obs) or self.T1w_obs/self.T2w_obs == 0 or math.isinf(self.T1w_obs/self.T2w_obs):
            raise Exception("Invalid T1/T2!")
        else: 
            popt, pcov, y_estimated = None, None, None
            if self.model_type == 0:
                if constrained:
                    popt, pcov = curve_fit(self.MT_model, freq, Zspec, p0=self.x0, 
                                           bounds=(self.lb, self.ub), method='trf', maxfev=5000)
                else:
                    popt, pcov = curve_fit(self.MT_model, freq, Zspec, p0=self.x0, 
                                       method='lm', maxfev=5000)
                self.fitted_paras = popt
                y_estimated = self.MT_model(freq, *popt)
            elif self.model_type == 1:
                if constrained:
                    popt, pcov = curve_fit(self.MT_model_1, freq, Zspec, p0=self.x0, 
                                           bounds=(self.lb, self.ub), method='trf', maxfev=5000)
                else:
                    popt, pcov = curve_fit(self.MT_model_1, freq, Zspec, p0=self.x0, 
                                       method='lm', maxfev=5000)
                self.fitted_paras = popt
                y_estimated = self.MT_model_1(freq, *popt)    
            elif self.model_type == 2:
                if constrained:
                    popt, pcov = curve_fit(self.MT_model_2, freq, Zspec, p0=self.x0, 
                                           bounds=(self.lb, self.ub), method='trf', maxfev=5000)
                else:
                    popt, pcov = curve_fit(self.MT_model_2, freq, Zspec, p0=self.x0, 
                                       method='lm', maxfev=5000)
                self.fitted_paras = popt
                y_estimated = self.MT_model_2(freq, *popt)                      
            return popt, y_estimated
      
    def cal_paras(self):
        if self.fitted_paras is None:
            raise Exception("No fitted parameters! Fit first.")
            
        if self.model_type == 0:
            R = self.fitted_paras[0]
            R_M0m_T1w = self.fitted_paras[1]
            T1w_T2w_ratio = self.fitted_paras[2]
            # following the formula
            numerator = R_M0m_T1w * (1/self.T1m - 1/self.T1w_obs)
            denominator = (1/self.T1m - 1/self.T1w_obs) + R
            T1w = self.T1w_obs * (1 + numerator/denominator)
            M0m = R_M0m_T1w / (R*T1w)
            T2w = T1w / T1w_T2w_ratio
            para_dict = {"R":R, "R*M0m*T1w":R_M0m_T1w, "T1w/T2w":T1w_T2w_ratio, 
                         "T2m":self.fitted_paras[3],
                         "M0m":M0m, "T1w":T1w, "T2w":T2w}
            return para_dict
        elif self.model_type == 1:
            R = self.fitted_paras[0]
            M0m_T1w = self.fitted_paras[1]
            T1w_T2w_ratio = self.fitted_paras[2]
            # following the formula
            numerator = R * M0m_T1w * (1/self.T1m - 1/self.T1w_obs)
            denominator = (1/self.T1m - 1/self.T1w_obs) + R
            T1w = self.T1w_obs * (1 + numerator/denominator)
            M0m = M0m_T1w / T1w
            T2w = T1w / T1w_T2w_ratio
            para_dict = {"R":R, "R*M0m*T1w":R*M0m_T1w, "T1w/T2w":T1w_T2w_ratio, 
                         "T2m":self.fitted_paras[3],
                         "M0m":M0m, "T1w":T1w, "T2w":T2w}
            return para_dict
        elif self.model_type == 2:
            R = self.R
            M0m_T1w = self.fitted_paras[0]
            T1w_T2w_ratio = self.fitted_paras[1]
            # following the formula
            numerator = R * M0m_T1w * (1/self.T1m - 1/self.T1w_obs)
            denominator = (1/self.T1m - 1/self.T1w_obs) + R
            T1w = self.T1w_obs * (1 + numerator/denominator)
            M0m = M0m_T1w / T1w
            T2w = T1w / T1w_T2w_ratio
            para_dict = {"R":R, "R*M0m*T1w":R*M0m_T1w, "T1w/T2w":T1w_T2w_ratio, 
                         "T2m":self.fitted_paras[2],
                         "M0m":M0m, "T1w":T1w, "T2w":T2w}
            return para_dict
   
    def generate_Zpsec(self, freq, paras):
        y = []
        if self.model_type == 0:
                return self.MT_model(freq, *paras)
        elif self.model_type == 1: 
                return self.MT_model_1(freq, *paras)        
        elif self.model_type == 2: 
                return self.MT_model_2(freq, *paras)

        
        
        
        
        
        