from unittest import TestCase

from go_mod_parser import extract_dependencies, retrieve_dependency, find_end_index, retrieve_dependencies_section


class Test(TestCase):
    def test_extract_dependencies(self):
        go_mod_content = """
require golang.org/x/net v1.2.3

require (
    golang.org/x/crypto v1.4.5 // indirect
    golang.org/x/text v1.6.7
)"""
        result = extract_dependencies(go_mod_content)
        self.assertEquals(result[0][0], 'golang.org/x/net')
        self.assertEquals(result[0][1], 'v1.2.3')
        self.assertEquals(result[1][0], 'golang.org/x/crypto')
        self.assertEquals(result[1][1], 'v1.4.5')
        self.assertEquals(result[2][0], 'golang.org/x/text')
        self.assertEquals(result[2][1], 'v1.6.7')

    def test_retrieve_dependency(self):
        line = 'require golang.org/x/net v1.2.3'
        library_name, library_version = retrieve_dependency(line)
        self.assertEquals(library_name, 'golang.org/x/net')
        self.assertEquals(library_version, 'v1.2.3')

    def test_find_end_index(self):
        lines = ['require golang.org/x/net v1.2.3',
                 '',
                 'require (',
                 'golang.org/x/crypto v1.4.5 // indirect',
                 'golang.org/x/text v1.6.7',
                 ')']
        end_index = find_end_index(lines, 2)
        self.assertEquals(end_index, 6)

    def test_retrieve_dependencies_section(self):
        lines = ['require golang.org/x/net v1.2.3',
                 '',
                 'require (',
                 'golang.org/x/crypto v1.4.5 // indirect',
                 'golang.org/x/text v1.6.7',
                 ')']
        result = retrieve_dependencies_section(lines, 2, 6)
        self.assertEquals(result[0][0], 'golang.org/x/crypto')
        self.assertEquals(result[0][1], 'v1.4.5')
        self.assertEquals(result[1][0], 'golang.org/x/text')
        self.assertEquals(result[1][1], 'v1.6.7')
