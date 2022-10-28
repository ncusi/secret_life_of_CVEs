from unittest import TestCase
from requirements_parser import extract_dependencies


class Test(TestCase):
    def test_extract_dependencies(self):
        lines = """
            bitmap==0.0.6
            cffi==1.12.3
            cycler==0.10.0
            Cython==0.29.12
        """
        result = extract_dependencies(lines)
        self.assertTrue(result[0][0] == 'bitmap')
        self.assertTrue(result[0][1] == [('==', '0.0.6')])
        self.assertTrue(result[1][0] == 'cffi')
        self.assertTrue(result[1][1] == [('==', '1.12.3')])
        self.assertTrue(result[2][0] == 'cycler')
        self.assertTrue(result[2][1] == [('==', '0.10.0')])
        self.assertTrue(result[3][0] == 'Cython')
        self.assertTrue(result[3][1] == [('==', '0.29.12')])
