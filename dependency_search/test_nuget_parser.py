from unittest import TestCase
from nuget_parser import extract_dependencies

class Test(TestCase):
    def test_extract_dependencies(self):
        xml_content = """<?xml version="1.0"?>
<package>
  <metadata>
    
    <id>test</id>
    <version>test</version>
    <title>test</title>
    <authors>test</authors>
    <owners>test</owners>
    <requireLicenseAcceptance>false</requireLicenseAcceptance>
    <description>test</description>
    <copyright>test</copyright>
    <tags>tag1 tag2</tags>
    
    <dependencies>
      <dependency id="Newtonsoft.Json" version="11.0.2" />
      <dependency id="RestSharp" version="106.3.1" />
      <dependency id="Selenium.Support" version="3.14.0" />
      <dependency id="Selenium.WebDriver" version="3.14.0" />
    </dependencies>
    
  </metadata>
</package>"""
        result = extract_dependencies(xml_content)
        self.assertTrue(result[0][0] == 'Newtonsoft.Json')
        self.assertTrue(result[0][1] == '11.0.2')
        self.assertTrue(result[1][0] == 'RestSharp')
        self.assertTrue(result[1][1] == '106.3.1')
        self.assertTrue(result[2][0] == 'Selenium.Support')
        self.assertTrue(result[2][1] == '3.14.0')
        self.assertTrue(result[3][0] == 'Selenium.WebDriver')
        self.assertTrue(result[3][1] == '3.14.0')


