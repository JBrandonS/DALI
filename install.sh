module purge && module load intel/oneAPI-2021
python3 -m venv --system-site-packages ~/.venv/cobaya
source ~/.venv/cobaya/bin/activate
pip3 install --upgrade pip setuptools
pip3 install --upgrade numba scb
pip3 install --upgrade cobaya
cobaya-install -p packages cosmo

cd packages/planck/code/plc_3.0/plc-3.1/
./waf clean
./waf configure --install_all_deps --lapack_mkl="$MKLROOT" --lapack_install
./waf install

mkdir -p chains/slurmdq