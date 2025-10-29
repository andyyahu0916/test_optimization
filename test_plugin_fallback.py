import unittest
import subprocess
import os
import configparser
import tempfile
import shutil

class TestPluginFallback(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory and a dummy config file."""
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, 'config.ini')

        # Create a dummy config file that points to an invalid plugin path
        config = configparser.ConfigParser()
        config['Simulation'] = {
            'simulation_time_ns': '1',
            'freq_charge_update_fs': '10',
            'freq_traj_output_ps': '100',
            'voltage': '1.0',
            'simulation_type': 'Constant_V',
            'platform': 'Reference',
            'mm_version': 'plugin',
            'logging_mode': 'legacy_print',
        }
        config['Validation'] = {
            'enable': 'false',
            'plugin_dir': '/path/to/non/existent/plugin/dir'
        }
        config['Files'] = {
            'outPath': os.path.join(self.test_dir, 'output/'),
            'ffdir': './ffdir/',
            'pdb_file': 'for_openmm.pdb',
            'residue_xml_list': 'sapt_residues.xml, graph_residue_c.xml, graph_residue_n.xml',
            'ff_xml_list': 'sapt_noDB_2sheets.xml, graph_c_freeze.xml, graph_n_freeze.xml'
        }
        config['Electrodes'] = {
            'cathode_index': '0,2',
            'anode_index': '1,3'
        }

        with open(self.config_path, 'w') as configfile:
            config.write(configfile)

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.test_dir)

    def test_run_openmm_exits_on_plugin_load_failure(self):
        """
        Verify that run_openMM.py exits with a non-zero status code
        when mm_version is 'plugin' but the plugin fails to load.
        """
        script_path = os.path.join('openMM_constant_V_beta', 'run_openMM.py')

        # We need to change the working directory so the script can find its dependencies
        working_dir = 'openMM_constant_V_beta'

        # To run the script from the correct directory, we adjust the path to the script and config
        adjusted_script_path = 'run_openMM.py'
        adjusted_config_path = self.config_path

        # Run the script as a subprocess
        process = subprocess.run(
            ['python', adjusted_script_path, '-c', adjusted_config_path],
            capture_output=True,
            text=True,
            cwd=working_dir
        )

        # 1. Assert that the process exited with a non-zero status code
        self.assertNotEqual(process.returncode, 0,
                            "run_openMM.py should have exited with a non-zero status code on plugin load failure.")

        # 2. Assert that a specific error message is present in stderr
        expected_error_message = "Error: No valid plugin directory found."
        self.assertIn(expected_error_message, process.stderr,
                      f"Expected error message '{expected_error_message}' not found in stderr.")

if __name__ == '__main__':
    unittest.main()
