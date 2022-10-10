from unittest import TestCase
from requirements_parser import extract_dependencies


class Test(TestCase):
    def test_extract_dependencies(self):
        lines = [
            'bitmap==0.0.6',
            'cffi==1.12.3',
            'cycler==0.10.0',
            'Cython==0.29.12'
        ]
        result = extract_dependencies(lines)
        self.assertTrue(result[0][0] == 'bitmap')
        self.assertTrue(result[0][1] == '0.0.6')
