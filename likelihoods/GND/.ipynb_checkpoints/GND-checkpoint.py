from cobaya.likelihood import Likelihood
from scipy.stats import gennorm
import numpy as np

class GND(Likelihood):

    def logp(self, _derived=None, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.
        """
        beta = params_values["beta"]
        x = params_values["x"]
        loc = params_values["loc"]
        scale = params_values["scale"]

        if _derived is not None:
            _derived["pdf"] = gennorm.pdf(x, beta, loc, scale)
            _derived["ppf"] = gennorm.ppf(x, beta, loc, scale)
        
        return gennorm.logpdf(x, beta, loc, scale)