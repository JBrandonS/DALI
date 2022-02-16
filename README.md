# [PArthENoPE](http://parthenope.na.infn.it/)
Public Algorithm Evaluating the Nucleosynthesis of Primordial Elements

PArthENoPE is a Fortran code for computing the evolution of abundances
of light nuclei during Big Bang Nucleosynthesis.
It also comes with a graphical user interface (GUI), written in python,
to simply the creation of input cards, running the code and producing
plots with the obtained outputs.

If you use the PArthENoPE 3.0 code, please cite the followint papers:  
**PArthENoPE Revolutions**  
_S. Gariazzo, P.F. de Salas, O. Pisanti, and R. Consiglio_  
to be published in Comput.Phys.Commun.
[arXiv:2103.05027](https://arxiv.org/abs/2103.05027)

**PArthENoPE reloaded**  
_R. Consiglio, P.F. de Salas, G. Mangano, G. Miele, S. Pastor, and O. Pisanti_  
Comput.Phys.Commun. 233 (2018) 237-242  
[DOI:10.1016/j.cpc.2018.06.022](https://doi.org/10.1016/j.cpc.2018.06.022)  
[arXiv:1712.04378](https://arxiv.org/abs/1712.04378)

**PArthENoPE: Public Algorithm Evaluating the Nucleosynthesis of Primordial
Elements**  
_Pisanti, O., A. Cirillo, S. Esposito, F. Iocco, G. Mangano, G. Miele, and
P.D. Serpico_  
Comput.Phys.Commun. 178 (2008) 956-971  
[DOI:10.1016/j.cpc.2008.02.015](https://doi.org/10.1016/j.cpc.2008.02.015)  
[arXiv:0705.0290](https://arxiv.org/abs/0705.0290)


# CONTENT OF THE PACKAGE

PArthENoPE3.0 package consists of the following files:

* a file `README.md` (this file) listing the content of the package and brief 
instructions on how to proceed.

* two FORTRAN files containing the main interface and program: `parthenope3.0.f` 
and `main3.0.f`.

* an auxiliary file `rates.dat` used by the FORTRAN code as input for the choice of
the main nuclear rates.

* four additional FORTRAN files containing the routines used by the main program: 
`addon3.0.f`, `dlsode.f`, `odepack1-parthenope3.0.f`, `odepack2-parthenope3.0.f`.

* a `Makefile` with some useful commands to compile the code, install the GUI
and launch some tests.

* an example configuration file `input3.0.card` (for compatibility reasons with
the previous version and to be used as input sample for testing the fortran code).

* sample output files, `parthenope3.0.out`, `nuclides3.0.out` and `info3.0.out`
obtained when the fortran code runs with the default parameters contained in 
`input3.0.card`.

* a number of Python files, the main executable `gui3.0.py` and the package
`parthenopegui/`, containing the code for running the GUI.

* two python utilities, `setup.py` and `setup.cfg`, to install the GUI dependencies.

* a script to facilitate the installation of the GUI within Anaconda environments,
`conda.sh`.

* a file `changelog3.0.txt` listing all changes made to PArthENoPE software.


# USE OF PARTHENOPE (in a nutshell)

If you want to use the standard version of PArthENoPE3.0:

1) Compile the FORTRAN source files using the `make` command. By default, the
Makefile is configured to use the `gfortran` compiler. If you have a different
one, you should either edit the `F90` option in the Makefile to indicate your
preference (recommended), or compile with `make F90=...`, where the name of your 
preferred compiler should replace the `...`. Notice, however, that the latter way 
does not allow the GUI to compile the executable, if it is missing. A modification 
of the `Makefile` is also required if the standard Fortran libraries are not 
available in the default paths: in such case, the user must edit the `FLIBS` 
variable appropriately.

2) Call the executable, `parthenope3.0` (interactive or card mode).

For installing and using the GUI:

1A) If you do not use Anaconda, you should be able to install the required python
dependencies with `make installguiuser` (recommended) or `make installgui` (within 
a virtual environment or in other cases). Add `sudo` in case superuser privileges 
are required. By default, the system `python` will be used. If you want to use a 
different python version, you should change the `PY` variable in the Makefile 
(recommended), or add `PY=...` to the previous commands, for example use `make 
installguiuser PY=python3` to install locally with `python3`.

1B) If you use Anaconda, we are aware of an incompatibility between the PySide2
version installable through the python repositories and the Qt version within
Anaconda, therefore you must not use the `PyPI` repositories. You should use
`make installconda` instead, that will create a new environment named
`parthenope3.0` and install the required packages by means of the `conda`
utilities and repositories. You can actually select the name for the new
environment by specifying `make installconda CONDAENV=...`, where your environment
name substitutes the `...`. Remember to activate your environment (`conda activate
parthenope3.0` or with the environment name of your choice) any time you want to
use the GUI.

2) Launch the GUI with the command: `make rungui`, or `python gui3.0.py &`. If you 
used a custom python version in the previous step, it is automatically adopted if 
you modified the Makefile and use `make rungui`, otherwise you can use `make rungui 
F90=...`. Otherwise, you should execute `gui3.0.py` explicitely with the same 
python version used to install, for example `python3 gui3.0.py &`.


# Contacts

Further information on PArthENoPE can be retrieved from the web page:
[http://parthenope.na.infn.it/](http://parthenope.na.infn.it/)

In case of problems, please contact [parthenope@na.infn.it](mailto:parthenope@na.infn.it)
