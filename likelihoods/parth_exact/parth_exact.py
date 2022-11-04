from cobaya.likelihood import Likelihood
import cobaya.log as log
from logging import DEBUG
import numpy as np
import pickle
from pathlib import Path
import numpy as np
from pyrthenope import Pyrthenope
class parth_exact(Likelihood):
    data_file: str
    parth_path: str
    tau_sigma: float
    obs_fid_errors: dict

    def initialize(self):
        super().initialize()
        self.logger = log.get_logger('parth_exact')
        self.nope = Pyrthenope(parthenope_path=Path(self.parth_path))
        
        with open(self.data_file, 'rb') as f:
            data = pickle.load(f)
            self.cosmoFid = data['cosmoFid']
            
        self.obs_fid_errors['tau'][1] = self.tau_sigma
    
    def get_requirements(self):
        return list(self.cosmoFid.keys())

    def logp(self, _derived=None, **kwargs):        
        self.nope.reset_card()
        
        for (k, v) in kwargs.items():
            self.logger.log(DEBUG, f'setting {k} to {v}')
            if k == 'N_nu':
                self.nope.card['DNNU'] = v - 3.045
            elif k == 'tau':
                self.nope.card['TAU'] = v
            elif k == 'eta10':
                self.nope.card['ETA10'] = v
            else:
                raise RuntimeError(f'Got unkown parameter {k} with value {v}')
                
        point = self.nope.run()
        self.logger.log(DEBUG, f'point: \n{point}')
        
        chi2 = 0
        for k,v in self.obs_fid_errors.items():
            self.logger.log(DEBUG, f'key: {k}, (fid): {v[0]}, point: {point[k].values[0]}, dchi2:{(point[k].values[0] - v[0])**2}')
            chi2 += (point[k].values[0] - v[0])**2 / (v[1]**2)
        return -chi2/2


