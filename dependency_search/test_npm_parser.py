from unittest import TestCase

from npm_parser import extract_dependencies


class Test(TestCase):
    def test_extract_dependencies(self):
        package_json_content = """{
  "name": "test-sample",
  "version": "0.1.0",
  "description": "test",
  "main": "index.js",
  "scripts": {
    "start": "node index.js"
  },
 "dependencies": { 
   "async": "^0.2.10", 
   "npm2es": "~0.4.2", 
   "optimist": "~0.6.0", 
   "request": "~2.30.0",
   "skateboard": "^1.5.1",
   "split": "^0.3.0",
   "weld": "^0.2.2"
  },
  "engines": {
    "node": "4.0.0"
  },
  "repository": {
    "type": "git",
    "url": "http://test_url"
  },
  "keywords": [
    "test"
  ],
  "author": "test",
  "contributors": [
    "test <test@test.test>"
  ],
  "license": "MIT"
}"""
        result = extract_dependencies(package_json_content)
        self.assertEqual(result[0][0], 'async')
        self.assertEqual(result[0][1], '^0.2.10')
        self.assertEqual(result[1][0], 'npm2es')
        self.assertEqual(result[1][1], '~0.4.2')
