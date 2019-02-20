import unittest
from ptp.utils.data_dict import DataDict

class TestDataDict(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestDataDict, self).__init__(*args, **kwargs)

        data_definitions = {
            'inputs': None,
            'targets': None
            }
        # Create object.
        self.data_dict = DataDict(data_definitions)

    def test_keys_present(self):
        """ Tests whether the original keys are present and can be retrieved/modified. """
        self.assertTrue('inputs' in self.data_dict.keys() )
        self.assertTrue('targets' in self.data_dict.keys() )

        # Check initial value.
        self.assertEqual(self.data_dict['inputs'], None)
        # Modify and retrieve.
        self.data_dict['inputs'] = 1.2
        self.assertEqual(self.data_dict['inputs'], 1.2)


    def test_keys_absent(self):
        """ Tests whether absent keys are really absent and cannot be simply added. """
        with self.assertRaises(KeyError):
            a = self.data_dict["predictions"]
        with self.assertRaises(KeyError):
            self.data_dict["predictions"] = 12


    def test_keys_extend(self):
        """ Tests whether append works as expected. """
        # Cannot add existing key.
        with self.assertRaises(KeyError):
            self.data_dict.extend( {"inputs": 1.5 } ) 
        # Can add new key.
        self.data_dict.extend( {"predictions": 12 } )
        self.assertEqual(self.data_dict['predictions'], 12)


if __name__ == '__main__':
    unittest.main()