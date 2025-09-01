import sysconfig, numpy
import os

print(sysconfig.get_paths()["include"])  # Python include dir in venv
print(sysconfig.get_config_var("LIBDIR"))  # Usually the folder with python311.lib
print(sysconfig.get_config_var("LDLIBRARY"))  # e.g., python311.lib
print(numpy.get_include())  # NumPy include dir
