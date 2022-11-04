from cobaya.likelihood import Likelihood
import cobaya.log as log
from logging import DEBUG, INFO
import numpy as np
import pickle

class dali(Likelihood):
    file: str
    spectrum: str
    experiment: int
    tau_prior: float

    def initialize(self):
        super().initialize()
        self.logger = log.get_logger('dali')
        try:
            with open(self.file, 'rb') as file:
                data = pickle.load(file)
        except Exception as err:
            raise log.LoggedError(self.logger, "Error reading file %s, %s", self.file, err)

        self.cosmoFid =  data['cosmoFid']
        # self.fisher = data['fisherGaussian'][self.experiment][self.spectrum]
        # self.dali3 = data['DALI3Gaussian'][self.experiment][self.spectrum]
        # self.dali4 = data['DALI4Gaussian'][self.experiment][self.spectrum]
        
        self.fisher = data['nfisherGaussian'][self.experiment][self.spectrum]
        self.dali3 = data['nDALI3Gaussian'][self.experiment][self.spectrum]
        self.dali4 = data['nDALI4Gaussian'][self.experiment][self.spectrum]
        
        self.logger.log(INFO, f'tau_prior: {self.tau_prior}')
        self.fisher[2,2] += 1/(self.tau_prior**2)
    
    def get_requirements(self):
        return self.cosmoFid.keys()

    def logp(self, _derived=None, **params_values):
        divVec = np.array([params_values[k] - v for k,v in self.cosmoFid.items()])
        
        logLike = -0.5*np.einsum('ij,i,j', self.fisher, divVec, divVec)
        logLike += -0.5*np.einsum('ijk,i,j,k', self.dali3, divVec, divVec, divVec)
        logLike += -0.125*np.einsum('ijkl,i,j,k,l', self.dali4, divVec, divVec, divVec, divVec)
            
        self.logger.log(DEBUG, f'params: {params_values} fids: {self.cosmoFid}')
        self.logger.log(DEBUG, f'divVec: {divVec} logLike: {logLike}')
        return logLike


