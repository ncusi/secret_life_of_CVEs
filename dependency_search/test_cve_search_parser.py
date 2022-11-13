from unittest import TestCase

import pandas as pd

from cve_search_parser import find_cve


class Test(TestCase):
    def test_find_cve(self):
        df = pd.DataFrame(['CVE-2014-2972', 'something CVE-2015-3972 something lese'], columns=['commit_message'])
        df = find_cve(df)
        self.assertEqual(df['cve'][0], 'CVE-2014-2972')
        self.assertEqual(df['cve'][1], 'CVE-2015-3972')
