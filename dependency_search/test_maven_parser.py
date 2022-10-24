from unittest import TestCase

from maven_parser import extract_dependencies, extract_dependencies_with_external_version, extract_properties, \
    prepare_artifact_url


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
        self.assertTrue(result[0][0] == 'junit')
        self.assertTrue(result[0][0] == 'junit')
        self.assertTrue(result[0][2] == '4.11')

    def test_extract_dependencies_on_version_from_properties(self):
        xml_content = """<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>test</groupId>
    <artifactId>test</artifactId>
    <version>1.0.0</version>
    <packaging>jar</packaging>


    <dependencies>
        <dependency>
            <groupId>org.openjdk.jmh</groupId>
            <artifactId>jmh-core</artifactId>
            <version>${jmh.version}</version>
        </dependency>
    </dependencies>

    <properties>
        <jmh.version>1.17.5</jmh.version>
    </properties>

</project>"""
        result = extract_dependencies(xml_content)
        self.assertTrue(result[0][0] == 'org.openjdk.jmh')
        self.assertTrue(result[0][1] == 'jmh-core')
        self.assertTrue(result[0][2] == '${jmh.version}')

    def test_extract_properties(self):
        xml_content = """<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>test</groupId>
    <artifactId>test</artifactId>
    <version>1.0.0</version>
    <packaging>jar</packaging>


    <dependencies>
        <dependency>
            <groupId>org.openjdk.jmh</groupId>
            <artifactId>jmh-core</artifactId>
            <version>${jmh.version}</version>
        </dependency>
    </dependencies>

    <properties>
        <jmh.version>1.17.5</jmh.version>
    </properties>

</project>"""
        result = extract_properties(xml_content)
        self.assertTrue(result['jmh.version'] == '1.17.5')

    def test_extract_dependencies_with_external_version(self):
        xml_content = """<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>test</groupId>
    <artifactId>test</artifactId>
    <version>1.0.0</version>
    <packaging>jar</packaging>


    <dependencies>
        <dependency>
            <groupId>org.openjdk.jmh</groupId>
            <artifactId>jmh-core</artifactId>
            <version>${jmh.version}</version>
        </dependency>
    </dependencies>

    <properties>
        <jmh.version>1.17.5</jmh.version>
    </properties>

</project>"""
        result = extract_dependencies_with_external_version(xml_content)
        self.assertTrue(result[0][0] == 'org.openjdk.jmh')
        self.assertTrue(result[0][1] == 'jmh-core')
        self.assertTrue(result[0][2] == '1.17.5')

    def test_prepare_artifact_url(self):
        dependencies = [('org.openjdk.jmh', 'jmh-core', '1.17.5')]
        result = prepare_artifact_url(dependencies)
        self.assertTrue(result[0][0] == 'org.openjdk.jmh')
        self.assertTrue(result[0][1] == 'jmh-core')
        self.assertTrue(result[0][2] == '1.17.5')
        self.assertTrue(result[0][3] == 'https://mvnrepository.com/artifact/org.openjdk.jmh/jmh-core/1.17.5')
