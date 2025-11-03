# setup.py for Cython compilation
"""
編譯 Cython 模組

使用方法:
    python setup_cython.py build_ext --inplace
    
編譯後會生成:
    electrode_charges_cython.so (Linux)
    electrode_charges_cython.pyd (Windows)
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "electrode_charges_cython",
        ["electrode_charges_cython.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=[
            "-O3",              # 最高優化等級
            "-march=native",    # 針對本機 CPU 優化
            "-ffast-math",      # 快速數學運算
        ],
        language="c",
    )
]

setup(
    name="electrode_charges_cython",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,      # 關閉邊界檢查
            'wraparound': False,       # 關閉負索引
            'cdivision': True,         # C 風格除法
            'initializedcheck': False, # 關閉初始化檢查
            'nonecheck': False,        # 關閉 None 檢查
        }
    ),
    zip_safe=False,
)
