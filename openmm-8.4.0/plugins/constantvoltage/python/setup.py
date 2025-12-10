"""
ConstantVoltage Plugin for OpenMM

This module provides:
- ConstantVoltageForce: Electrode force with SCF charge updates
- ConstantVDrudeLangevinIntegrator: Drude Langevin dynamics with dual-temperature thermostat

Installation:
    pip install .

Usage:
    from openmm.constantvoltage import ConstantVoltageForce, ConstantVDrudeLangevinIntegrator
    
    # Create force
    cv_force = ConstantVoltageForce()
    cv_force.setVoltage(1.0)  # 1V
    cv_force.setLgap(2.0)     # 2nm gap
    cv_force.setLcell(4.0)    # 4nm cell
    
    # Add electrode atoms
    for i, area in electrode_atoms:
        cv_force.addCathodeAtom(i, area)
    
    # Add to system
    system.addForce(cv_force)
    
    # Create integrator
    integrator = ConstantVDrudeLangevinIntegrator(300, 1.0, 1, 40.0, 0.001)
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import os
import sys

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}'
        ]
        
        build_args = ['--config', 'Release']
        
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
            
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

setup(
    name='openmm-constantvoltage',
    version='1.0.0',
    author='Andy',
    author_email='andy@example.com',
    description='ConstantVoltage Plugin for OpenMM - Fixed voltage electrode simulations',
    long_description=open('README.md').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
    url='https://github.com/openmm/openmm',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Chemistry',
    ],
    python_requires='>=3.8',
    install_requires=['openmm>=8.0'],
    ext_modules=[CMakeExtension('openmm.constantvoltage')],
    cmdclass={'build_ext': CMakeBuild},
    packages=['openmm'],
    package_dir={'openmm': '.'},
    zip_safe=False,
)
