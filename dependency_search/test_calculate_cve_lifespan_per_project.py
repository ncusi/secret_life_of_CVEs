from unittest import TestCase

import pandas as pd

from calculate_cve_lifespan_per_project import calculate_commits_per_cve, calculate_dep_per_cve_df, calculate_embargo, \
    calculate_lifespan


def prepare_example_df():
    df = pd.DataFrame.from_dict({'commit': {1249: '0de7239e563eff6e83c3e72d7deb9fd26a54a3a7',
                                            1251: '93a32bda3388f419f281d5e7f44687ecb132a654',
                                            1252: '93a32bda3388f419f281d5e7f44687ecb132a654',
                                            1253: '161e4419056c54aca8fa924de00efe5da8269a31',
                                            4411: '1d0b005096d65aaf4f2ecd2fb319a9d1dab46b55'},
                                 'commit_cves': {1249: 'CVE-2014-2972',
                                                 1251: 'CVE-2014-2972',
                                                 1252: 'CVE-2014-2957',
                                                 1253: 'CVE-2014-2972',
                                                 4411: 'CVE-2014-2972'},
                                 'commiter_time': {1249: '2014-07-21 10:28:07',
                                                   1251: '2014-07-22 11:39:44',
                                                   1252: '2014-07-22 11:39:44',
                                                   1253: '2014-07-22 16:03:59',
                                                   4411: '2014-07-22 14:23:17'},
                                 'author_time': {1249: '2014-07-18 14:42:08',
                                                 1251: '2014-07-22 11:39:44',
                                                 1252: '2014-07-22 11:39:44',
                                                 1253: '2014-07-22 16:03:59',
                                                 4411: '2014-07-22 14:23:17'},
                                 'project_names': {1249: 'Exim_exim',
                                                   1251: 'pfsense_FreeBSD-ports',
                                                   1252: 'pfsense_FreeBSD-ports',
                                                   1253: 'salsa.debian.org_security-tracker-team_security-tracker',
                                                   4411: 'git-portage_git-portage'},
                                 'total_number_of_files': {1249: 4, 1251: 5, 1252: 5, 1253: 1, 4411: 3},
                                 'published_date': {1249: '2014-09-04T17:55:00',
                                                    1251: '2014-09-04T17:55:00',
                                                    1252: '2014-09-04T17:55:00',
                                                    1253: '2014-09-04T17:55:00',
                                                    4411: '2014-09-04T17:55:00'},
                                 'error': {1249: None, 1251: None, 1252: None, 1253: None, 4411: None},
                                 'other_languages': {1249: 0.0, 1251: 0.0, 1252: 0.0, 1253: 0.0, 4411: 2.0},
                                 'used_dep_manager': {1249: False,
                                                      1251: False,
                                                      1252: False,
                                                      1253: False,
                                                      4411: False},
                                 'lang_Ada': {1249: 4, 1251: 5, 1252: 0, 1253: 0, 4411: 0},
                                 'lang_Assembly': {1249: 0, 1251: 0, 1252: 0, 1253: 0, 4411: 0},
                                 'lang_C': {1249: 0, 1251: 0, 1252: 5, 1253: 1, 4411: 3}})
    return df


class Test(TestCase):
    def test_calculate_commits_per_cve(self):
        df = prepare_example_df()
        result_df = calculate_commits_per_cve(df)
        self.assertEqual(result_df.shape, (5, 3))
        self.assertEqual(result_df['project_names'][0], 'Exim_exim')
        self.assertEqual(result_df['project_names'][1], 'git-portage_git-portage')
        self.assertEqual(result_df['project_names'][2], 'pfsense_FreeBSD-ports')
        self.assertEqual(result_df['project_names'][3], 'pfsense_FreeBSD-ports')
        self.assertEqual(result_df['project_names'][4], 'salsa.debian.org_security-tracker-team_security-tracker')

    def test_calculate_language_files_per_cve(self):
        df = prepare_example_df()
        result_df = calculate_commits_per_cve(df)
        self.assertEqual(result_df['project_names'][0], 'Exim_exim')
        self.assertEqual(result_df['project_names'][1], 'git-portage_git-portage')
        self.assertEqual(result_df['project_names'][2], 'pfsense_FreeBSD-ports')
        self.assertEqual(result_df['project_names'][3], 'pfsense_FreeBSD-ports')
        self.assertEqual(result_df['project_names'][4], 'salsa.debian.org_security-tracker-team_security-tracker')

    def test_calculate_lifespan(self):
        df = prepare_example_df()
        result_df = calculate_lifespan(df)
        self.assertEqual(result_df['project_names'][0], 'pfsense_FreeBSD-ports')
        self.assertEqual(result_df['project_names'][1], 'Exim_exim')
        self.assertEqual(result_df['project_names'][2], 'git-portage_git-portage')
        self.assertEqual(result_df['project_names'][3], 'pfsense_FreeBSD-ports')
        self.assertEqual(result_df['project_names'][4], 'salsa.debian.org_security-tracker-team_security-tracker')

    def test_calculate_embargo(self):
        df = prepare_example_df()
        result_df = calculate_embargo(df)
        self.assertEqual(result_df['project_names'][0], 'Exim_exim')
        self.assertEqual(result_df['project_names'][1], 'git-portage_git-portage')
        self.assertEqual(result_df['project_names'][2], 'pfsense_FreeBSD-ports')
        self.assertEqual(result_df['project_names'][3], 'pfsense_FreeBSD-ports')
        self.assertEqual(result_df['project_names'][4], 'salsa.debian.org_security-tracker-team_security-tracker')

    def test_calculate_dep_per_cve_df(self):
        df = prepare_example_df()
        result_df = calculate_dep_per_cve_df(df)
        self.assertEqual(result_df['project_names'][0], 'Exim_exim')
        self.assertEqual(result_df['project_names'][1], 'git-portage_git-portage')
        self.assertEqual(result_df['project_names'][2], 'pfsense_FreeBSD-ports')
        self.assertEqual(result_df['project_names'][3], 'pfsense_FreeBSD-ports')
        self.assertEqual(result_df['project_names'][4], 'salsa.debian.org_security-tracker-team_security-tracker')
