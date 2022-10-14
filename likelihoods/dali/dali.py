from cobaya.likelihood import Likelihood
from cobaya.log import LoggedError
import numpy as np
import pickle

class dali(Likelihood):
    file: str
    use_dali: bool
    spectrum: str
    experiment: int
    remove: list

    def initialize(self):
        try:
            with open(self.file, 'rb') as file:
                data = pickle.load(file)
        except Exception as err:
            raise LoggedError(self.log, "Error reading file %s, %s", self.file, err)

        self.cosmoFid =  data['cosmoFid']
        self.fisher = data['fisherGaussian'][self.experiment][self.spectrum]
        if self.use_dali:
            self.dali3 = data['DALI3Gaussian'][self.experiment][self.spectrum]
            self.dali4 = data['DALI4Gaussian'][self.experiment][self.spectrum]

        # removes requested parameters from dali and fisher
        if self.remove:
            for k in self.remove:
                try:
                    i = list(self.cosmoFid.keys()).index(k)
                    self.cosmoFid.pop(k)
                    self.fisher = np.delete(np.delete(self.fisher, i, 0), i, 1)
                    if self.use_dali:
                        self.dali3 = np.delete(np.delete(np.delete(self.dali3, i, 0), i, 1), i, 2)
                        self.dali4 = np.delete(np.delete(np.delete(np.delete(self.dali4, i, 0), i, 1), i, 2), i, 3)
                except ValueError:
                    raise LoggedError(self.log, "Could not find parameter %s to remove", k)

        return super().initialize()
    
    def get_requirements(self):
        return [i for i in self.cosmoFid.keys() if i not in self.remove]

    def logp(self, _derived=None, **params_values):
        divVec = np.array([params_values[k] - v for k,v in self.cosmoFid.items()])
        logLike = -0.5*np.einsum('ij,i,j', self.fisher, divVec, divVec)
        if self.use_dali:
            logLike += -0.5*np.einsum('ijk,i,j,k', self.dali3, divVec, divVec, divVec)
            logLike += -0.125*np.einsum('ijkl,i,j,k,l', self.dali4, divVec, divVec, divVec, divVec)
        return logLike


