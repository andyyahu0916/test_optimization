from setuptools import setup, Extension
import os
import platform

openmm_dir = '/home/andy/miniforge3/envs/cuda'
constantvplugin_header_dir = '/home/andy/test_optimization/openMM_constantV_plugin/ConstantVPlugin/openmmapi/include'
constantvplugin_library_dir = '/home/andy/test_optimization/openMM_constantV_plugin/ConstantVPlugin/build'

# setup extra compile and link arguments on Mac
extra_compile_args = ['-std=c++11']
extra_link_args = []

if platform.system() == 'Darwin':
    extra_compile_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7']
    extra_link_args += ['-stdlib=libc++', '-mmacosx-version-min=10.7', '-Wl', '-rpath', openmm_dir+'/lib']

extension = Extension(name='_constantvplugin',
                      sources=['ConstantVPluginWrapper.cpp'],
                      libraries=['OpenMM', 'ConstantVPlugin'],
                      include_dirs=[os.path.join(openmm_dir, 'include'), constantvplugin_header_dir],
                      library_dirs=[os.path.join(openmm_dir, 'lib'), constantvplugin_library_dir],
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_link_args
                     )

setup(name='constantvplugin',
      version='1.0',
      py_modules=['constantvplugin'],
      ext_modules=[extension],
     )
