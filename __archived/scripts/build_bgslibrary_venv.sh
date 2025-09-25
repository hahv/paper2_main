cd /home/ha/dev/
git clone --recursive https://github.com/andrewssobral/bgslibrary.git
cd bgslibrary

source /home/ha/dev/bgslibrary-examples-python/.venv/bin/activate
cd /home/ha/dev/bgslibrary
rm -rf build  # Clear previous build to avoid cache issues
mkdir build && cd build
cmake .. \
  -DPYTHON_EXECUTABLE=/home/ha/dev/bgslibrary-examples-python/.venv/bin/python \
  -DBGS_CORE_STATIC=ON \
  -DBGS_PYTHON_SUPPORT=ON \
  -DBGS_PYTHON_ONLY=ON \
  -DBGS_PYTHON_VERSION=3.11 \
  -DNUMPY_INCLUDE_DIR=$(python -c "import numpy; print(numpy.get_include())") \
  -Dpybind11_DIR=/home/ha/dev/bgslibrary/modules/pybind11 \
  -DCMAKE_INSTALL_PREFIX=/home/ha/dev/bgslibrary-examples-python/.venv
make -j8
make install