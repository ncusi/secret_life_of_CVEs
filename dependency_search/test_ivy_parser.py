from unittest import TestCase

from ivy_parser import extract_dependencies


class Test(TestCase):
    def test_extract_dependencies(self):
        xml_content = """<ivy-module version="2.0">
    <info organisation="test" module="test"/>
    <dependencies>
        <dependency org="commons-lang" name="commons-lang" rev="2.0"/>
        <dependency org="commons-cli" name="commons-cli" rev="1.0"/>
    </dependencies>
</ivy-module>"""
        result = extract_dependencies(xml_content)
        self.assertEqual(result[0][0], 'commons-lang')
        self.assertEqual(result[0][1], 'commons-lang')
        self.assertEqual(result[0][2], '2.0')
        self.assertEqual(result[1][0], 'commons-cli')
        self.assertEqual(result[1][1], 'commons-cli')
        self.assertEqual(result[1][2], '1.0')
