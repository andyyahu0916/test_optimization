from setuptools import setup, Extension
import os
import platform

openmm_dir = '@OPENMM_DIR@'
electrodecharge_header_dir = '@ELECTRODECHARGE_HEADER_DIR@'
electrodecharge_library_dir = '@ELECTRODECHARGE_LIBRARY_DIR@'

extra_compile_args = ['-std=c++11']
extra_link_args = []

if platform.system() == 'Darwin':
    extra_compile_args += ['-stdlib=libc++', '-mmacosx-version-min=10.9']
    extra_link_args += ['-stdlib=libc++', '-mmacosx-version-min=10.9', '-Wl', '-rpath', os.path.join(openmm_dir, 'lib')]

extension = Extension(
    name='_electrodecharge',
    sources=['ElectrodeChargePluginWrapper.cpp'],
    libraries=['OpenMM', 'ElectrodeChargePlugin'],
    include_dirs=[os.path.join(openmm_dir, 'include'), electrodecharge_header_dir],
    library_dirs=[os.path.join(openmm_dir, 'lib'), electrodecharge_library_dir],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    name='electrodecharge',
    version='2.0',
    py_modules=['electrodecharge'],
    ext_modules=[extension],
)
