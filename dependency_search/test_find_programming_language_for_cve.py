from unittest import TestCase

from find_programming_language_for_cve import get_languages, add_language_to_df
import pandas as pd


class Test(TestCase):
    def test_get_languages(self):
        column_name = 'ext_.nasm'
        extension_to_language = {
            '.nasm': ['Assembly'],
        }
        languages = get_languages(column_name, extension_to_language)
        self.assertIn('Assembly', languages)

    def test_add_language_to_df(self):
        df = pd.DataFrame(data={
            'commit': ['80cfab8f','84967c'],
            'ext_.nasm':[1,None],
            'ext_.c':[None,3]}
        )
        extension_to_language = {
            '.nasm': ['Assembly'],
            '.c': ['C']
        }
        result_df = add_language_to_df(df, extension_to_language)
        self.assertEqual(result_df['commit'][0],'80cfab8f')
        self.assertEqual(result_df['lang_Assembly'][0],1)
        self.assertEqual(result_df['commit'][1],'84967c')
        self.assertEqual(result_df['lang_C'][1],3)
