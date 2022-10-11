from unittest import TestCase

from maven_parser import extract_dependencies


class Test(TestCase):
    def test_extract_dependencies(self):
        xml_content = """<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
	xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
	<modelVersion>4.0.0</modelVersion>

	<groupId>pl.umk.wmii.msr.contributions</groupId>
	<artifactId>msr14</artifactId>
	<version>0.0.1-SNAPSHOT</version>
	<packaging>jar</packaging>

	<name>msr14</name>
	<url>http://maven.apache.org</url>

	<properties>
		<project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
		<maven.compiler.source>1.8</maven.compiler.source>
		<maven.compiler.target>1.8</maven.compiler.target>
	</properties>

	<dependencies>
		<dependency>
			<groupId>junit</groupId>
			<artifactId>junit</artifactId>
			<version>4.11</version>
			<scope>test</scope>
		</dependency>
    </dependencies>
</project>"""
        result = extract_dependencies(xml_content)
        self.assertTrue(result[0][0] == 'junit:junit')
        self.assertTrue(result[0][1] == '4.11')
