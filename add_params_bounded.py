import sys
import pickle
import numpy as np
import json
import yaml

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
    'eta10':r'\eta_{10}',
}

def add_priors(yaml_file, data_file, tau):
    params = {}
    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    fid = data['cosmoFid']
    
    #errors = {'omega_b_h2': 0.00022, 'N_nu': 0.36, 'tau': 0.5}
    errors = {'eta10': 273*0.00022, 'N_nu': 0.36, 'tau': 0.5}
    widths = 5

    for i, (k, v) in enumerate(fid.items()):
        val = {'ref': v, 'latex': latex_trans.get(k, k), 'prior': {'min': v-widths*errors[k], 'max':v+widths*errors[k]}}
        params.update({k: val})

    with open(yaml_file, 'r') as f:
        idata = yaml.safe_load(f)
        idata.update(eval(json.dumps({'params': params})))
       
    print('Adding to ', yaml_file)
    print('From ', data_file)
    print('Final Data ', idata)
        
    with open(yaml_file, 'w') as f:
        yaml.dump(idata, f, default_flow_style=False)

print('...Finding params')
add_priors(sys.argv[1], sys.argv[2], sys.argv[3])
print('...done')