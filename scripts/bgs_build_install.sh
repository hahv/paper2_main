source /mnt/e/NextCloud/paper2_main/.venv/bin/python
cd /home/ha/dev/bgslibrary
rm -rf build # Clear previous build to avoid cache issues
mkdir build && cd build
cmake .. \
	-DPYTHON_EXECUTABLE=/mnt/e/NextCloud/paper2_main/.venv/bin/python \
	-DBGS_CORE_STATIC=ON \
	-DBGS_PYTHON_SUPPORT=ON \
	-DBGS_PYTHON_ONLY=ON \
	-DBGS_PYTHON_VERSION=3.11 \
	-DNUMPY_INCLUDE_DIR=$(python -c "import numpy; print(numpy.get_include())") \
	-Dpybind11_DIR=/home/ha/dev/bgslibrary/modules/pybind11 \
	-DCMAKE_INSTALL_PREFIX=/mnt/e/NextCloud/paper2_main/.venv
make -j8
make install

