""" Omid Sani, Shanechi Lab, University of Southern California, 2016 """
# pylint: disable=C0103, C0111

"Tests the matHelper module"

import unittest
import MatHelper as mh
import numpy as np

class TestIOHelper(unittest.TestCase):

    def isEqualValue(self, fD, data):
        if isinstance(data, dict):
            for var, value in data.items():
                self.isEqualValue(fD[var], data[var])
        elif type(data) is np.ndarray:
            np.testing.assert_array_equal(fD, data)
        elif type(data) is list:
            for i in range(len(data)):
                self.isEqualValue(fD[i], data[i])
        else:
            self.assertEqual(fD, data)


    def test_io_from_mat(self):
        # Make sample dictionary
        settings = {
            'do_this': True,
            'do_that': False,
            'innerSettings': {
                'inner_ok': True,
                'inner_array': np.array([1, 2]),
                'inner_number': 10,
                'NoneVal': None,
                'strList': ['ABC', 'A'] # This will be loaded as 'ABC', 'A  '
            }
        }
        data = {
            'data': np.array([1, 2, 3]),
            'time': np.array([0, 1, 2]),
            'Fs': 1, 
            'settings': settings,
            'perfStats': [
                {'a': 1, 'b': 0.5},
                {'a': 2, 'b': 1}
            ]
        }
        filePath = './testFile.mat'
        mh.savemat(filePath, data)
        fD = mh.loadmat(filePath)
        dataFix = mh.replaceNone(data, np.array([]))
        self.isEqualValue(fD, dataFix)


    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()