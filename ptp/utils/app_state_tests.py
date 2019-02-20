import unittest
from ptp.utils.app_state import AppState

class TestAppState(unittest.TestCase):


    def test_01keys_present(self):
        """ Tests whether the original keys are present and can be retrieved/modified. """
        # Initialize object.
        app_state = AppState()
        # Add global.
        app_state["global1"] = 1 
        # Check its value.
        self.assertEqual(app_state['global1'], 1)

    def test_02keys_present_singleton(self):
        """ Tests whether the original keys are still present in new AppState "instance". """
        # Initialize object.
        app_state = AppState()
        # Check its value.
        self.assertEqual(app_state['global1'], 1)

    def test_03keys_absent(self):
        """ Tests whether absent keys are really absent. """
        with self.assertRaises(KeyError):
            a = AppState()["global2"]

    def test_04keys_overwrite(self):
        """ Tests whether you can overwrite existing key. """
        with self.assertRaises(KeyError):
            AppState()["global1"] = 2


if __name__ == '__main__':
    unittest.main()