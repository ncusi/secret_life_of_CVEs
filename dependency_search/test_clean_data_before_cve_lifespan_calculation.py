from unittest import TestCase

import pandas as pd

from clean_data_before_cve_lifespan_calculation import remove_additional_languages, remove_incorrect_dates, \
    remove_more_than_one_project


class Test(TestCase):
    def test_remove_additional_languages(self):
        df = pd.DataFrame(data={
            'commit': ['80cfab8f', '84967c'],
            'commit_cves': ['CVE-2014-2972', 'CVE-2014-2972'],
            'commiter_time': ['2014-08-04 15:38:27', '2014-07-23 11:44:25'],
            'author_time': ['2014-08-04 15:34:55', '2014-07-23 11:44:25'],
            'project_names': ['mirror-rpm_exim', 'buildroot_buildroot'],
            'total_number_of_files': [0, 3052],
            'published_date': ['2014-09-04T17:55:00', '2014-09-04T17:55:00'],
            'error': [None, None],
            'ext_.nasm': [1, None],
            'ext_.c': [None, 3],
            'lang_C': [None, 3]}
        )
        result_df = remove_additional_languages(df, ['lang_C'])
        self.assertEqual(result_df.shape, (1, 9))

    def test_remove_incorrect_dates(self):
        df = pd.DataFrame(data={
            'commit': ['d26201', '3c02d100a', '857d28'],
            'commit_cves': ['CVE-2008-5079', 'CVE-2017-18342', 'CVE-2012-1668'],
            'commiter_time': ['2000-05-23 21:15:00', '2030-04-09 03:16:53', '2015-05-08 11:42:05'],
            'author_time': ['2000-05-23 21:15:0', '2030-04-09 03:16:53', '2015-05-08 11:42:05'],
            'project_names': ['broftkd_linux-history-repo', 'desperax_generator', 'pagodabox_nanobox-pkgsrc-lite'],
            'total_number_of_files': [21, 89, 82966],
            'published_date': ['2008-12-09T00:30:00', '2018-06-27T12:29:00', None],
            'error': [None, None, None],
            'lang_C': [1, 3, 4]}
        )
        result_df = remove_incorrect_dates(df)
        self.assertEqual(result_df.shape, (1, 9))

    def test_remove_more_than_one_project(self):
        df = pd.DataFrame(data={
            'commit': ['d26201', 'd06eddd15'],
            'commit_cves': ['CVE-2008-5079', 'CVE-2014-3470'],
            'commiter_time': ['2000-05-23 21:15:00', '2014-06-20 16:17:41'],
            'author_time': ['2000-05-23 21:15:0', '2014-06-20 15:00:00'],
            'project_names': ['broftkd_linux-history-repo', 'bloomberg_chromium.bb;grpc_grpc'],
            'total_number_of_files': [21, 1],
            'published_date': ['2008-12-09T00:30:00', '2014-06-05T21:55:00'],
            'error': [None, None],
            'lang_C': [1, 4]}
        )
        result_df = remove_more_than_one_project(df)
        self.assertEqual(result_df.shape, (1, 9))
