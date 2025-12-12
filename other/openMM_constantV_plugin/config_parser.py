#!/usr/bin/env python3
"""
Configuration Parser for ConstantV Plugin

Reads .ini configuration files and provides easy access to all parameters.
"""

import os
import sys
import configparser
from typing import List, Tuple


class SimulationConfig:
    """
    解析并存储模拟配置参数
    """

    def __init__(self, config_file: str):
        """
        从.ini文件加载配置

        Parameters:
        -----------
        config_file : str
            配置文件路径
        """
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"配置文件不存在: {config_file}")

        self.config = configparser.ConfigParser()
        self.config.read(config_file)

        # 验证必需的sections
        required_sections = ['System', 'Electrodes', 'SCF', 'Simulation', 'Output', 'Platform']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"配置文件缺少section: [{section}]")

        # 解析所有参数
        self._parse_system()
        self._parse_electrodes()
        self._parse_scf()
        self._parse_electrolyte()
        self._parse_simulation()
        self._parse_output()
        self._parse_platform()
        self._parse_advanced()

    def _parse_system(self):
        """解析[System] section"""
        section = self.config['System']

        self.pdb_file = section.get('pdb_file')
        if not os.path.exists(self.pdb_file):
            raise FileNotFoundError(f"PDB文件不存在: {self.pdb_file}")

        self.forcefield_dir = section.get('forcefield_dir')
        if not os.path.exists(self.forcefield_dir):
            raise FileNotFoundError(f"Force field目录不存在: {self.forcefield_dir}")

        # 解析force field文件列表
        ff_files_str = section.get('forcefield_files', '')
        self.forcefield_files = [
            os.path.join(self.forcefield_dir, f.strip())
            for f in ff_files_str.split(',')
            if f.strip()
        ]

        # 验证所有force field文件存在
        for ff_file in self.forcefield_files:
            if not os.path.exists(ff_file):
                raise FileNotFoundError(f"Force field文件不存在: {ff_file}")

        self.nonbonded_cutoff = section.getfloat('nonbonded_cutoff', 1.4)

    def _parse_electrodes(self):
        """解析[Electrodes] section"""
        section = self.config['Electrodes']

        self.voltage = section.getfloat('voltage', 0.0)

        # 解析cathode chain indices
        cathode_str = section.get('cathode_chains', '0')
        self.cathode_chains = tuple(
            int(x.strip()) for x in cathode_str.split(',') if x.strip()
        )

        # 解析anode chain indices
        anode_str = section.get('anode_chains', '1')
        self.anode_chains = tuple(
            int(x.strip()) for x in anode_str.split(',') if x.strip()
        )

        # 解析排除元素
        exclude_str = section.get('exclude_elements', '')
        self.exclude_elements = tuple(
            x.strip() for x in exclude_str.split(',') if x.strip()
        )

    def _parse_scf(self):
        """解析[SCF] section"""
        section = self.config['SCF']

        self.num_scf_iterations = section.getint('num_iterations', 4)
        self.scf_frequency_fs = section.getfloat('scf_frequency_fs', 200.0)

    def _parse_electrolyte(self):
        """解析[Electrolyte] section"""
        if 'Electrolyte' in self.config:
            section = self.config['Electrolyte']
            self.natom_cutoff = section.getint('natom_cutoff', 100)
        else:
            self.natom_cutoff = 100

    def _parse_simulation(self):
        """解析[Simulation] section"""
        section = self.config['Simulation']

        self.total_time_ns = section.getfloat('total_time_ns', 0.5)
        self.timestep_ps = section.getfloat('timestep_ps', 0.001)
        self.temperature = section.getfloat('temperature', 300.0)
        self.temperature_drude = section.getfloat('temperature_drude', 1.0)
        self.friction_per_ps = section.getfloat('friction_per_ps', 1.0)
        self.friction_drude_per_ps = section.getfloat('friction_drude_per_ps', 1.0)

    def _parse_output(self):
        """解析[Output] section"""
        section = self.config['Output']

        self.output_dir = section.get('output_dir', 'output')
        self.trajectory_output_ps = section.getfloat('trajectory_output_ps', 10.0)
        self.log_output_steps = section.getint('log_output_steps', 100)
        self.write_charges = section.getboolean('write_charges', False)
        self.overwrite_output = section.getboolean('overwrite_output', True)

    def _parse_platform(self):
        """解析[Platform] section"""
        section = self.config['Platform']

        self.platform_name = section.get('platform_name', 'CUDA')
        self.cuda_precision = section.get('cuda_precision', 'mixed')

    def _parse_advanced(self):
        """解析[Advanced] section（可选）"""
        if 'Advanced' in self.config:
            section = self.config['Advanced']
            self.sapt_ff_exclusions = section.getboolean('sapt_ff_exclusions', True)
            self.constraints = section.get('constraints', 'HBonds')
            self.rigid_water = section.getboolean('rigid_water', True)
            self.recursion_limit = section.getint('recursion_limit', 2000)
            self.console_output_frequency_ps = section.getfloat('console_output_frequency_ps', 10.0)
        else:
            self.sapt_ff_exclusions = True
            self.constraints = 'HBonds'
            self.rigid_water = True
            self.recursion_limit = 2000
            self.console_output_frequency_ps = 10.0

    def calculate_scf_frequency_steps(self) -> int:
        """
        计算SCF频率（以steps为单位）

        Returns:
        --------
        int : SCF频率（多少步做一次SCF）
        """
        timestep_fs = self.timestep_ps * 1000  # ps -> fs
        if timestep_fs <= 0:
            raise ValueError("时间步长必须为正数")

        steps = int(self.scf_frequency_fs / timestep_fs)
        return max(1, steps)

    def calculate_trajectory_output_steps(self) -> int:
        """
        计算轨迹输出频率（以steps为单位）

        Returns:
        --------
        int : 轨迹输出频率（多少步输出一次）
        """
        timestep_fs = self.timestep_ps * 1000  # ps -> fs
        return int(self.trajectory_output_ps * 1000 / timestep_fs)

    def calculate_total_steps(self) -> int:
        """
        计算总步数

        Returns:
        --------
        int : 总步数
        """
        timestep_fs = self.timestep_ps * 1000  # ps -> fs
        return int(self.total_time_ns * 1e6 / timestep_fs)

    def calculate_console_output_steps(self) -> int:
        """
        计算console输出频率（以steps为单位）

        Returns:
        --------
        int : console输出频率
        """
        timestep_fs = self.timestep_ps * 1000  # ps -> fs
        return int(self.console_output_frequency_ps * 1000 / timestep_fs)

    def get_constraints_enum(self):
        """
        获取OpenMM的constraints枚举

        Returns:
        --------
        OpenMM constraints enum
        """
        from openmm import app

        constraints_map = {
            'HBonds': app.HBonds,
            'AllBonds': app.AllBonds,
            'None': None
        }

        return constraints_map.get(self.constraints, app.HBonds)

    def print_summary(self):
        """打印配置摘要"""
        print("="*70)
        print("模拟配置摘要")
        print("="*70)

        print("\n[系统文件]")
        print(f"  PDB: {self.pdb_file}")
        print(f"  Force fields: {len(self.forcefield_files)}个文件")
        print(f"  Nonbonded cutoff: {self.nonbonded_cutoff} nm")

        print("\n[电极配置]")
        print(f"  电压: {self.voltage} V")
        print(f"  Cathode chains: {self.cathode_chains}")
        print(f"  Anode chains: {self.anode_chains}")
        print(f"  排除元素: {self.exclude_elements}")

        print("\n[SCF参数]")
        print(f"  迭代次数: {self.num_scf_iterations}")
        print(f"  SCF频率: 每{self.scf_frequency_fs} fs ({self.calculate_scf_frequency_steps()}步)")

        print("\n[模拟参数]")
        print(f"  总时间: {self.total_time_ns} ns ({self.calculate_total_steps()}步)")
        print(f"  时间步长: {self.timestep_ps} ps")
        print(f"  温度: {self.temperature} K")
        print(f"  Drude温度: {self.temperature_drude} K")
        print(f"  摩擦系数: {self.friction_per_ps} 1/ps")
        print(f"  Drude摩擦: {self.friction_drude_per_ps} 1/ps")

        print("\n[输出设置]")
        print(f"  输出目录: {self.output_dir}")
        print(f"  轨迹输出: 每{self.trajectory_output_ps} ps ({self.calculate_trajectory_output_steps()}步)")
        print(f"  Log输出: 每{self.log_output_steps}步")
        print(f"  写出电荷: {self.write_charges}")

        print("\n[计算平台]")
        print(f"  平台: {self.platform_name}")
        if self.platform_name == 'CUDA':
            print(f"  精度: {self.cuda_precision}")

        print("="*70)


def load_config(config_file: str = 'simulation_config.ini') -> SimulationConfig:
    """
    便捷函数：加载配置文件

    Parameters:
    -----------
    config_file : str
        配置文件路径，默认为'simulation_config.ini'

    Returns:
    --------
    SimulationConfig : 配置对象
    """
    try:
        config = SimulationConfig(config_file)
        return config
    except Exception as e:
        print(f"✗ 错误: 无法加载配置文件: {e}")
        sys.exit(1)


if __name__ == '__main__':
    """测试配置解析器"""
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = 'simulation_config.ini'

    print(f"加载配置文件: {config_file}\n")

    try:
        config = SimulationConfig(config_file)
        config.print_summary()

        print("\n✓ 配置文件解析成功!")

    except Exception as e:
        print(f"\n✗ 配置文件解析失败: {e}")
        sys.exit(1)
