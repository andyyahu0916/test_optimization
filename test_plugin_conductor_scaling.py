import unittest
import subprocess
import os
import configparser
import tempfile
import shutil
import numpy as np

class TestPluginConductorScaling(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()
        self.original_config_path = os.path.join(self.test_dir, 'config_original.ini')
        self.plugin_config_path = os.path.join(self.test_dir, 'config_plugin.ini')

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.test_dir)

    def _create_config_file(self, filepath, mm_version):
        """Helper function to create a config.ini file for the test."""
        config = configparser.ConfigParser()
        config['Simulation'] = {
            'simulation_time_ns': '0.001',
            'freq_charge_update_fs': '10',
            'freq_traj_output_ps': '1',
            'voltage': '1.0',
            'simulation_type': 'Constant_V',
            'platform': 'Reference',
            'mm_version': mm_version,
            'logging_mode': 'efficient',
            'write_charges': 'True',
        }
        config['Files'] = {
            'outPath': os.path.join(self.test_dir, f'output_{mm_version}/'),
            'ffdir': './ffdir/',
            'pdb_file': 'for_openmm.pdb',
            'residue_xml_list': 'sapt_residues.xml, graph_residue_c.xml, graph_residue_n.xml',
            'ff_xml_list': 'sapt_noDB_2sheets.xml, graph_c_freeze.xml, graph_n_freeze.xml'
        }
        config['Electrodes'] = {
            'cathode_index': '0,2',
            'anode_index': '1,3'
        }
        with open(filepath, 'w') as configfile:
            config.write(configfile)

    def _run_simulation_and_get_charges(self, config_path, mm_version):
        """Run the simulation and parse the output charges.dat file."""
        script_path = 'run_openMM.py'
        working_dir = 'openMM_constant_V_beta'

        # Correctly get the mm_version from the config path to construct the output path
        mm_version = os.path.basename(config_path).replace('config_', '').replace('.ini', '')
        output_dir = os.path.join(working_dir, self.test_dir, f'output_{mm_version}')

        process = subprocess.run(
            ['python', script_path, '-c', config_path],
            capture_output=True, text=True, cwd=working_dir
        )

        self.assertEqual(process.returncode, 0, f"Simulation failed for {config_path}. Stderr:\n{process.stderr}")

        charges_path = os.path.join(output_dir, 'charges.dat')

        self.assertTrue(os.path.exists(charges_path), f"charges.dat not found at {charges_path}")

        # Read the last line of charges
        with open(charges_path, 'r') as f:
            lines = f.readlines()
            last_line = lines[-1].strip()

        return np.fromstring(last_line, sep=' ')

    def test_conductor_scaling_equivalence(self):
        """
        Verify that the plugin's conductor scaling is equivalent to the Python implementation.
        """
        # 1. Create config files
        self._create_config_file(self.original_config_path, 'original')
        self._create_config_file(self.plugin_config_path, 'plugin')

        # 2. Run both simulations and get charges
        charges_original = self._run_simulation_and_get_charges(self.original_config_path, 'original')
        charges_plugin = self._run_simulation_and_get_charges(self.plugin_config_path, 'plugin')

        # 3. Compare the charges
        self.assertEqual(charges_original.shape, charges_plugin.shape, "Charge arrays have different shapes.")

        # Conductor atoms are the last ones in our simple PDB
        conductor_charge_original = charges_original[-1]
        conductor_charge_plugin = charges_plugin[-1]

        # Using a tolerance to account for floating point differences
        tolerance = 1e-5
        self.assertAlmostEqual(conductor_charge_original, conductor_charge_plugin,
                               delta=tolerance,
                               msg=f"Conductor charges are not equivalent. Original: {conductor_charge_original}, Plugin: {conductor_charge_plugin}")

if __name__ == '__main__':
    unittest.main()
