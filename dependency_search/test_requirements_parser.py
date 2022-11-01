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
        self.assertEquals(result[0][0], 'bitmap')
        self.assertEquals(result[0][1], [('==', '0.0.6')])
        self.assertEquals(result[1][0], 'cffi')
        self.assertEquals(result[1][1], [('==', '1.12.3')])
        self.assertEquals(result[2][0], 'cycler')
        self.assertEquals(result[2][1], [('==', '0.10.0')])
        self.assertEquals(result[3][0], 'Cython')
        self.assertEquals(result[3][1], [('==', '0.29.12')])
