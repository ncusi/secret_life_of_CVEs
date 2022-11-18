from unittest import TestCase

import pandas as pd

from cve_search_parser import find_cve
from cve_search_parser import find_file_name_extensions


class Test(TestCase):
    def test_find_cve(self):
        df = pd.DataFrame(['CVE-2014-2972', 'something CVE-2015-3972 something lese'], columns=['commit_message'])
        df = find_cve(df)
        self.assertEqual(df['cve'][0], 'CVE-2014-2972')
        self.assertEqual(df['cve'][1], 'CVE-2015-3972')

    def test_find_file_name_extensions(self):
        changed_file_names = [b'Dockerfile', b'README.md', b'docker-compose.yml', b'img/exec_evil.png',
                              b'img/exploit.png', b'img/tmp_list.png', b'requirements.txt', b'vuln/.DS_Store',
                              b'vuln/CVE-2014-0472/.DS_Store', b'vuln/CVE-2014-0472/__init__.py',
                              b'vuln/CVE-2014-0472/asgi.py', b'vuln/CVE-2014-0472/settings.py',
                              b'vuln/CVE-2014-0472/urls.py', b'vuln/CVE-2014-0472/wsgi.py', b'vuln/app/.DS_Store',
                              b'vuln/app/__init__.py', b'vuln/app/admin.py', b'vuln/app/apps.py',
                              b'vuln/app/migrations/__init__.py',
                              b'vuln/app/migrations/__pycache__/__init__.cpython-37.pyc', b'vuln/app/models.py',
                              b'vuln/app/tests.py', b'vuln/app/views.py', b'vuln/db.sqlite3', b'vuln/manage.py',
                              b'vuln/run.sh']
        extensions = find_file_name_extensions(changed_file_names)
        self.assertEqual(extensions['.md'], 1)
        self.assertEqual(extensions['.yml'], 1)
        self.assertEqual(extensions['.png'], 3)
        self.assertEqual(extensions['.txt'], 1)
        self.assertEqual(extensions['.py'], 13)
        self.assertEqual(extensions['.pyc'], 1)
        self.assertEqual(extensions['.sqlite3'], 1)
        self.assertEqual(extensions['.sh'], 1)
