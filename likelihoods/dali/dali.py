from cobaya.likelihood import Likelihood
import cobaya.log as log
from logging import DEBUG
import numpy as np
import pickle

class dali(Likelihood):
    file: str
    use_dali: bool
    spectrum: str
    experiment: int
    remove: list
    tau_prior: float

    def initialize(self):
        self.logger = log.get_logger('dali')
        try:
            with open(self.file, 'rb') as file:
                data = pickle.load(file)
        except Exception as err:
            raise log.LoggedError(self.logger, "Error reading file %s, %s", self.file, err)

        self.cosmoFid =  data['cosmoFid']
        self.fisher = data['fisherGaussian'][self.experiment][self.spectrum]
        self.logger.log(DEBUG, f'Fisher before: {self.fisher}')
        self.fisher[2,2] += 1/self.tau_prior**2
        self.logger.log(DEBUG, f'Fisher after: {self.fisher}')
        
        if self.use_dali:
            self.dali3 = data['DALI3Gaussian'][self.experiment][self.spectrum]
            self.dali4 = data['DALI4Gaussian'][self.experiment][self.spectrum]

        return super().initialize()
    
    def get_requirements(self):
        return [i for i in self.cosmoFid.keys() if i not in self.remove]

    def logp(self, _derived=None, **params_values):
        divVec = np.array([params_values[k] - v for k,v in self.cosmoFid.items()]) # should this be abundances??
        logLike = -0.5*np.einsum('ij,i,j', self.fisher, divVec, divVec)
        if self.use_dali:
            logLike += -0.5*np.einsum('ijk,i,j,k', self.dali3, divVec, divVec, divVec)
            logLike += -0.125*np.einsum('ijkl,i,j,k,l', self.dali4, divVec, divVec, divVec, divVec)
            
        self.logger.log(DEBUG, f'params: {params_values} fids: {self.cosmoFid}')
        self.logger.log(DEBUG, f'divVec: {divVec} logLike: {logLike}')
        return logLike


