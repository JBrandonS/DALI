from cobaya.likelihood import Likelihood
from cobaya.log import LoggedError
from cobaya.model import get_model

import math
import numpy as np
import pickle

class cmb(Likelihood):
    noise_file: str
    b_modes: bool
    lensing_extraction: bool
    unlensed_clTTTEEE: bool
    f_sky: float
    l_min: int
    l_max: int
    exp_number: int
    fiducial_params: dict

    def initialize(self):
        return super().initialize()

    def initialize_with_provider(self, provider):
        super().initialize_with_provider(provider)

        info_fiducial = provider.model._updated_info
        info_fiducial['params'] = self.fiducial_params
        info_fiducial['likelihood'] = {'one': None}

        model_fiducial = get_model(info_fiducial)
        model_fiducial.add_requirements({"Cl": {'tt': self.l_max, 'te': self.l_max, 'ee': self.l_max, 'bb': self.l_max, 'pp': self.l_max}})
        model_fiducial.logposterior({})
        
        self.Cls = model_fiducial.provider.get_Cl(ell_factor=False, units="muK2")

        try:
            with open(self.noise_file, 'rb') as file:
                data_noise = pickle.load(file)
        except Exception as err:
            raise LoggedError(self.log, "Error reading noise file %s, %s", self.noise_file, err)

        self.noise_T = data_noise['cmbNoiseSpectra'][self.exp_number]['cl_TT'][:self.l_max+1]
        self.noise_P = data_noise['cmbNoiseSpectra'][self.exp_number]['cl_EE'][:self.l_max+1]
        # default:
        numCls = 3
        # deal with BB:
        if self.b_modes:
            self.index_B = numCls
            numCls += 1
        # deal with pp (p = CMB lensing potential):
        if self.lensing_extraction:
            self.index_pp = numCls
            numCls += 1
            # read the NlDD noise (noise for the extracted deflection field spectrum)
            self.Nldd = data_noise['deflectionNoises'][self.exp_number][:self.l_max+1]

        # initialize the fiducial values
        self.Cl_est = np.zeros((numCls, self.l_max+1), 'float64')
        self.Cl_est[0] = self.Cls['tt'][:self.l_max + 1]
        self.Cl_est[2] = self.Cls['te'][:self.l_max + 1]
        self.Cl_est[1] = self.Cls['ee'][:self.l_max + 1]
        # BB:
        if self.b_modes:
            self.Cl_est[self.index_B] = self.Cls['bb'][:self.l_max + 1]
        # DD (D = deflection field):
        if self.lensing_extraction:
            self.Cl_est[self.index_pp] = self.Cls['pp'][:self.l_max + 1]
    
    def get_requirements(self):
        used = ['te', 'ee', 'bb', 'pp']
        return {"Cl": {cl: self.l_max for cl in used}}

    def logp(self, **params_values):
        # get Cl's from the cosmological code
        Cl_theo = dict()
        cltd_fid = 0
        cltd = 0

        # if we want unlensed Cl's
        if self.unlensed_clTTTEEE:
            Cl_theo['tt'] = self.provider.get_unlensed_Cl(ell_factor=False, units="muK2")['tt'][:self.l_max+1]
            Cl_theo['te'] = self.provider.get_unlensed_Cl(ell_factor=False, units="muK2")['te'][:self.l_max+1]
            Cl_theo['ee'] = self.provider.get_unlensed_Cl(ell_factor=False, units="muK2")['ee'][:self.l_max+1]
            if self.lensing_extraction:
                Cl_theo['pp'] = self.provider.get_Cl(ell_factor=False, units="muK2")['pp'][:self.l_max+1] # lensing potential power spectrum is always unitless
            if self.b_modes:
                Cl_theo['bb'] = self.provider.get_unlensed_Cl(ell_factor=False, units="muK2")['bb'][:self.l_max+1]

        # if we want lensed Cl's
        else:
            Cl_theo['tt'] = self.provider.get_Cl(ell_factor=False, units="muK2")['tt'][:self.l_max+1]
            Cl_theo['te'] = self.provider.get_Cl(ell_factor=False, units="muK2")['te'][:self.l_max+1]
            Cl_theo['ee'] = self.provider.get_Cl(ell_factor=False, units="muK2")['ee'][:self.l_max+1]
            if self.lensing_extraction:
                Cl_theo['pp'] = self.provider.get_Cl(ell_factor=False, units="muK2")['pp'][:self.l_max+1] # lensing potential power spectrum is always unitless
            if self.b_modes:
                Cl_theo['bb'] = self.provider.get_Cl(ell_factor=False, units="muK2")['bb'][:self.l_max+1]

        # compute likelihood

        chi2 = 0

        # cound number of modes.
        # number of modes is different form number of spectra
        # modes = T,E,[B],[D=deflection]
        # spectra = TT,EE,TE,[BB],[DD,TD]
        # default:
        num_modes=2
        # add B mode:
        if self.b_modes:
            num_modes += 1
        # add D mode:
        if self.lensing_extraction:
            num_modes += 1

        Cov_est = np.zeros((num_modes, num_modes), 'float64')
        Cov_the = np.zeros((num_modes, num_modes), 'float64')
        Cov_mix = np.zeros((num_modes, num_modes), 'float64')

        for l in range(self.l_min, self.l_max+1):

            # case with B modes (priority over LensingExtraction):
            if self.b_modes:
                Cov_est = np.array([
                    [self.Cl_est[0, l], self.Cl_est[2, l], 0],
                    [self.Cl_est[2, l], self.Cl_est[1, l], 0],
                    [0, 0,self.Cl_est[3, l]]])
                Cov_the = np.array([
                    [Cl_theo['tt'][l]+self.noise_T[l], Cl_theo['te'][l], 0],
                    [Cl_theo['te'][l], Cl_theo['ee'][l]+self.noise_P[l], 0],
                    [0, 0, Cl_theo['bb'][l]+self.noise_P[l]]])

            # case with lensing
            # note that the likelihood is base on ClDD (deflection spectrum)
            # rather than Clpp (lensing potential spectrum)
            # But the Bolztmann code input is Clpp
            # So we make the conversion using ClDD = l*(l+1.)*Clpp
            elif self.lensing_extraction:

                cldd_fid = l*(l+1.)*self.Cl_est[self.index_pp, l]
                cldd = l*(l+1.)*Cl_theo['pp'][l]

                Cov_est = np.array([
                    [self.Cl_est[0, l], self.Cl_est[2, l], 0],
                    [self.Cl_est[2, l], self.Cl_est[1, l], 0],
                    [cltd_fid, 0, cldd_fid]])
                Cov_the = np.array([
                    [Cl_theo['tt'][l]+self.noise_T[l], Cl_theo['te'][l], 0],
                    [Cl_theo['te'][l], Cl_theo['ee'][l]+self.noise_P[l], 0],
                    [cltd, 0, cldd+self.Nldd[l]]])

            # case without B modes nor lensing:
            else:
                Cov_est = np.array([
                    [self.Cl_est[0, l], self.Cl_est[2, l]],
                    [self.Cl_est[2, l], self.Cl_est[1, l]]])
                Cov_the = np.array([
                    [Cl_theo['tt'][l]+self.noise_T[l], Cl_theo['te'][l]],
                    [Cl_theo['te'][l], Cl_theo['ee'][l]+self.noise_P[l]]])

            # get determinant of observational and theoretical covariance matrices
            det_est = np.linalg.det(Cov_est)
            det_the = np.linalg.det(Cov_the)

            # get determinant of mixed matrix (= sum of N theoretical
            # matrices with, in each of them, the nth column replaced
            # by that of the observational matrix)
            det_mix = 0.
            for i in range(num_modes):
                Cov_mix = np.copy(Cov_the)
                Cov_mix[:, i] = Cov_est[:, i]
                det_mix += np.linalg.det(Cov_mix)

            chi2 += (2.*l+1.)*self.f_sky *\
                (det_mix/det_the + math.log(det_the/det_est) - num_modes)

        return -chi2/2


