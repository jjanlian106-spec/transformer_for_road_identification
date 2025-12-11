import matlab.engine
import os

my_engine = matlab.engine.start_matlab()
env_name = "/Users/lzj/Desktop/Data/ATask/file/Matlab_studyfile/simu_py.slx"
my_engine.load_system(env_name)

# --- EDIT THIS: set to the directory that contains your simu_py_use.m ---
# Example: mfile_dir = "/Users/lzj/Desktop/Data/ATask/file/Matlab_studyfile/mfiles"
mfile_dir = "/Users/lzj/Desktop/Data/ATask/file/Matlab_studyfile"

# Add the directory (and subdirectories) to MATLAB path so the .m function is found
my_engine.addpath(my_engine.genpath(mfile_dir), nargout=0)

# Alternatively, you can change MATLAB's current folder instead of adding to path:
# my_engine.cd(mfile_dir, nargout=0)

# Call the MATLAB function (do NOT include the .m extension)
my_engine.simu_py_use(nargout=0)

