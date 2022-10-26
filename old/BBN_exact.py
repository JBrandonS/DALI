from distutils.log import debug
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math

from pyrthenope import *

from cobaya.run import run
from getdist.mcsamples import MCSamplesFromCobaya
import getdist.plots as gdplt

from numpy import loadtxt, savetxt, mgrid, argwhere, c_, arange, linspace, reshape
from sympy import symbols, diff, lambdify
from itertools import combinations

from matplotlib.axes import Axes

import sys

t_input = sys.argv[1] if sys.argv[1] else 0.5
print('using sigma_tau =', t_input)

gdplot = gdplt.get_subplot_plotter()

job = 'bbn_full_t'+str(t_input)
exact_file = 'exact_data_t'+str(t_input)+'.pkl'

# [gparams + removeParams] has to be the complete list of parameters
# [gparams] should be leq size of observables
gparams = ['omega_b_h2', 'N_nu', 'tau']

experiment = 0
spectrum = 'unlensed'
priorWidth = 5

parth_inout_map = {
    'DNNU': 'N_nu',
    'ETA10': 'eta10',
    'TAU': 'tau',
    'omega_b_h2': 'OmegaBh^2',
}

card_mod = {
    'OUTPUT': 'F  2  3 6'
}

latex_trans = {
    'omega_c_h2':r'\Omega_\mathrm{cdm}h^2', 
    'omega_b_h2':r'\Omega_b h^2', 
    'N_eff':r'N_\mathrm{eff}',
    'N_nu':r'N_\mathrm{eff}', # Just for the effect
    'A_s':r'A_\mathrm{s}', 
    'n_s':r'n_\mathrm{s}', 
    'theta_s':r'\theta_\mathrm{s}',
    'tau':r'\tau_n\;[s]',
    'mnu':r'm_\mathrm{\nu}',
    'eta10':r'\eta_\mathrm{10}'
}

def get_data(file=job + '.pkl', experiment=experiment, spectrum=spectrum, prior_width=priorWidth):
    params = {}
    with open(file, 'rb') as f:
        data = pickle.load(f)

    fid = data['cosmoFid']
    fisher = data['fisherGaussian'][experiment][spectrum]
    is2 = data['iSigma2']

    inv = np.linalg.inv(fisher)
    errors = np.sqrt(np.diag(inv))
    widths = prior_width * errors

    for i, (k, v) in enumerate(fid.items()):
        val = {'ref': v, 'latex': latex_trans.get(k, k), 'prior': {'min': v-widths[i], 'max':v+widths[i]}}
        params.update({k: val})
    return params, fid, inv, is2

def get_exact(file=job + '.pkl', experiment=experiment, spectrum=spectrum, prior_width=priorWidth):
    params, cosmoFid, inv, is2 = get_data(file, experiment, spectrum, prior_width)   

    def _exact(**kwargs):
        # \sum_a (f_a(p)-f_{a,fid))^2 / ( 2 \sigma^2(f_a) )
        nope = Pyrthenope(card_mod=card_mod)        
        for (k, v) in kwargs.items():
            if k == 'omega_b_h2':
                nope.card['TAU'] = nope.omega_to_eta(v)
            elif k == 'N_nu':
                nope.card['DNNU'] = v - 3.045
            elif k == 'tau':
                nope.card['TAU'] = v
            else:
                nope.card[k] = v
        point = nope.run()

        chi2 = 0
        for i,(k,v) in enumerate(kwargs.items()):
            chi2 += (point[parth_inout_map.get(k, k)].values[0] - cosmoFid[k])**2 * is2.get(i,i) / 2
        return -chi2
    
    model = {
        'likelihood': {
            'exact': { 
                'external': _exact,
                'input_params': list(params.keys()),
                'stop_at_error': True,
            }
        },
        'params': params,
        'sampler': {
            'mcmc': {
                'oversample_power': 0.4,
                'proposal_scale': 1.9,
                'Rminus1_stop': 0.05,
                'Rminus1_cl_stop': 0.05,
                'max_samples': 100,
            }
        },
        'debug': True
    }
    return model

# Setup a default model
updatedExact, samplerExact = run(get_exact())
gdsExact = MCSamplesFromCobaya(updatedExact, samplerExact.products()["sample"], name_tag='Exact')

with open(exact_file, 'wb') as file:
    pickle.dump(gdsExact, file, protocol=pickle.HIGHEST_PROTOCOL)