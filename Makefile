# fortran compiler:
F90 ?= gfortran
# python executable:
PY ?= python
#edit FLIBS if you need to specify a custom path for the libraries
FLIBS=
# for Anaconda users, the name of the environment that will be created:
CONDAENV ?=

### the rest of the Makefile should not need to be modified ###
FOPTS=-O3

OBJECTS=parthenope3.0.f main3.0.f dlsode.f odepack1-parthenope3.0.f odepack2-parthenope3.0.f addon3.0.f
EXENAME=parthenope3.0

default: all

all: parthenope3.0

parthenope3.0: $(OBJECTS)
	$(F90) $(OBJECTS) -o $(EXENAME) $(FLIBS) $(FOPTS)

clean:
	rm parthenope3.0

installconda:
	bash conda.sh $(CONDAENV)

installguiuser:
	$(PY) setup.py install --user

installgui:
	$(PY) setup.py install

rungui:
	$(PY) gui3.0.py &

testgui:
	$(PY) parthenopegui/tests.py
