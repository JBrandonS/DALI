from cobaya.likelihood import Likelihood
import cobaya.log as log
from logging import DEBUG
import numpy as np
import pickle
from pathlib import Path

from pyrthenope import *

class parth_exact(Likelihood):
    data_file: str
    parth_path: str
    tau_sigma: float
    obs_fid_errors: dict

    def initialize(self):
        self.logger = log.get_logger('parth_exact')
        self.nope = Pyrthenope(parthenope_path=Path(self.parth_path))
        
        with open(self.data_file, 'rb') as f:
            data = pickle.load(f)
            self.cosmoFid = data['cosmoFid']
            
        # self.obs_fid_errors['tau'][1] = self.tau_sigma
        
        self.sigma2 = 0  
        for v in self.obs_fid_errors.values():
            self.sigma2 += (float(v[1]))**2
        # self.sigma2 = self.sigma2 / len(self.obs_fid_errors)
            
        return super().initialize()
    
    def get_requirements(self):
        return list(self.cosmoFid.keys())
    
    def omega_to_eta10(self, omega: float, yp = 0.2449, t0 = 2.7255):
        return 273.279 * omega / (1 - 0.007125 * yp) * (2.7255 / t0)**3

    def logp(self, _derived=None, **kwargs):        
        self.nope.reset_card()
        
        for (k, v) in kwargs.items():
            self.logger.log(DEBUG, f'setting {k} to {v}')
            if k == 'omega_b_h2':
                self.nope.card['ETA10'] = self.omega_to_eta10(v)
            elif k == 'N_nu':
                self.nope.card['DNNU'] = v - 3.045
            elif k == 'tau':
                self.nope.card['TAU'] = v
            elif k == 'eta10':
                self.nope.card['ETA10'] = v
            else:
                raise RuntimeError(f'Got unkown parameter {k} with value {v}')
                
        self.logger.log(DEBUG, f'running card: \n{self.nope.card}')
        point = self.nope.run()
        self.logger.log(DEBUG, f'point: \n{point}')
        
        chi2 = 0
        for k,v in self.obs_fid_errors.items():
            self.logger.log(DEBUG, f'key: {k}, (fid): {v[0]}, point: {point[k].values[0]}, dchi2:{(point[k].values[0] - float(v[0]))**2}')
            chi2 += (point[k].values[0] - float(v[0]))**2 / (2 * float(v[1])**2)
        return -chi2


