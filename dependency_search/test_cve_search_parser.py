from unittest import TestCase

import pandas as pd

from cve_search_parser import add_cve, extract_cve, find_file_name_dependencies, find_file_name_documentation, \
    find_file_name_extensions, find_most_common_extension


class Test(TestCase):
    def test_add_cve(self):
        df = pd.DataFrame(
            [['80cfab8fdefa20cef32e5e591ebf9bc47d1d7bc5', 'CVE-2014-2972', 1407180895, 1407180895],
             ['8011cd56e39a433b1837465259a9bd24a38727fb', 'something CVE-2015-3972 something lese', 1407180895,
              1407180895]],
            columns=['commit', 'commit_message', 'commiter_time', 'author_time'])
        result_df = add_cve(df)
        self.assertEqual(result_df['commit_cves'][0], ['CVE-2014-2972'])
        self.assertEqual(result_df['commit_cves'][1], ['CVE-2015-3972'])

    def test_extract_cve(self):
        commit = '80cfab8fdefa20cef32e5e591ebf9bc47d1d7bc5',
        commit_message = 'something CVE-2014-2972 and another cve 2012 4405'
        commit_time = 1407180895
        author_time = 1407180895
        result = extract_cve(commit, commit_message, commit_time, author_time)
        self.assertEqual(result.get('commit'), commit)
        self.assertEqual(result.get('commit_cves'), ['CVE-2014-2972', 'CVE-2012-4405'])
        self.assertEqual(result.get('commiter_time'), '2014-08-04 21:34:55')
        self.assertEqual(result.get('author_time'), '2014-08-04 21:34:55')

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
        self.assertEqual(extensions['ext_.md'], 1)
        self.assertEqual(extensions['ext_.yml'], 1)
        self.assertEqual(extensions['ext_.png'], 3)
        self.assertEqual(extensions['ext_.txt'], 1)
        self.assertEqual(extensions['ext_.py'], 13)
        self.assertEqual(extensions['ext_.pyc'], 1)
        self.assertEqual(extensions['ext_.sqlite3'], 1)
        self.assertEqual(extensions['ext_.sh'], 1)

    def test_find_dependency_files(self):
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
        documentation = find_file_name_documentation(changed_file_names)
        self.assertEqual(documentation['doc_readme_count'], 1)

    def test_find_file_name_dependencies(self):
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
        documentation = find_file_name_dependencies(changed_file_names)
        self.assertEqual(documentation['dep_requirements'], 1)
        self.assertEqual(documentation['dep_maven'], 0)

    def test_find_most_common_extension(self):
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
        most_common_extension = find_most_common_extension(extensions)
        self.assertEqual(most_common_extension, '.py')

    def test_find_most_common_extension_when_multiple_files_have_same_number_of_occurrences(self):
        changed_file_names = [b'Dockerfile',
                              b'README.md',
                              b'docker-compose.yml',
                              b'vuln/run.sh']
        extensions = find_file_name_extensions(changed_file_names)
        most_common_extension = find_most_common_extension(extensions)
        self.assertEqual(most_common_extension, ';.md;.yml;.sh')