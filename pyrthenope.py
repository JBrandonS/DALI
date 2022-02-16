import os
import subprocess
import pandas as pd
import numpy as np
import atexit
import copy
import tempfile
from typing import Dict, Any
from pathlib import Path

class Pyrthenope(object):
    """ Python wrapper for parthenope v3.0

    see: 
        https://arxiv.org/pdf/0705.0290.pdf  for a good description for the interal and options
        https://arxiv.org/pdf/1712.04378.pdf for the update for 2.0 but mostly talks about the modifications
        https://arxiv.org/pdf/2103.05027.pdf for the current version, but mostly talks about the GUI
    """
    
    def __init__(self, card_mod = None, parthenope_path: Path = Path.cwd()) -> None:
        self._tmpDir = tempfile.TemporaryDirectory()
        self.tmpPath = Path(self._tmpDir.name)

        self._default_card = {
            # see 0705.0290 table V for some of the allowed ranges
            'TAU': 879.4,                   # experimental value of neutron lifetime
            'DNNU': 0.,                     # number of extra neutrinos
            'XIE': 0.,                      # nu_e chemical potential
            'XIX': 0.,                      # nu_x chemical potential
            'RHOLMBD': 0.,                  # value of cosmological constant at the BBN epoch
            'ETA10': 6.13832,               # value of eta10
            'NETWORK': 26,                  # number of nuclides in the network
    #       'RATES': '1  (20 2 2.)',        # optional rates, see 0705.0290
            'OVERWRITE': 'F',               # option for overwriting the output files
            'FOLLOW': 'F',                  # option for following the evolution on the screen
            'OUTPUT': 'T  4  1 2 3 6',      # options for customizing the output, see 0705.0290 table 1 and page 9
            'FILES': 'parthenope.out nuclides.out info.out'
        }

        self.card = copy.deepcopy(self._default_card)
        if card_mod is not None:
            self.modify_card(card_mod, True)

        self.parth_exe = parthenope_path / 'parthenope3.0'
        self.file = self.tmpPath / 'temp.card'
        self.parth_out_file = self.tmpPath / 'parthenope.out'
        atexit.register(self.exit)

    def exit(self):
        self._tmpDir.cleanup()

    def modify_card(self, mod: Dict[str, Any], apply_to_default: bool = False):
        for k, v in mod.items():
            self.card[k] = v
            if apply_to_default:
                self._default_card[k] = v

    def reset_card(self):
        self.card = copy.deepcopy(self._default_card)

    def get_data(self, file: Path, fix_gui_data: bool = False):
        data = pd.read_csv(file, delim_whitespace=True)

        # Since our data using fortran double precsion we need to cast everything into np.float64
        # On some versions of python, numpy is not able to use D as a double precision float so we change it
        # CHECK: That all outputs are floats / ints and the replace doesnt mess anything up
        data = data.applymap(lambda x: np.float64(str(x).replace('D', 'E')))

        # this is only needed for data generated by the GUI
        # since the header starts with a #, pandas reads in the first col as '#', we shift all the names to remove this and remove
        #   the last column which will just be NaN since the data doesnt exist
        if fix_gui_data:
            cols = data.columns[1:]
            data = data.drop(data.columns[-1], axis='columns')
            data = data.rename({o:n for o, n in zip(data.columns, cols)}, axis='columns')

        return data

    def run(self, std_out: int = subprocess.DEVNULL):
        with open(self.file, mode="x") as c:
            for k, v in self.card.items():
                c.write(f"{k} {v}\n")
            c.write('EXIT')

        subprocess.run([self.parth_exe], 
                        input=f"c\n{self.file}\n",
                        encoding="ascii",
                        cwd = self.tmpPath, 
                        stdout=std_out)
        return self.get_data(self.parth_out_file)


