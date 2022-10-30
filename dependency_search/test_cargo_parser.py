from unittest import TestCase
from cargo_parser import extract_dependencies


class Test(TestCase):
    def test_extract_dependencies(self):
        lines = """
[dependencies]
bincode = "1.3.1"
serde = { version = "1.0", features = ["derive"] }
        """
        result = extract_dependencies(lines)
        self.assertTrue(result[0][0] == 'bincode')
        self.assertTrue(result[0][1] == '1.3.1')
        self.assertTrue(result[1][0] == 'serde')
        self.assertTrue(result[1][1] == '1.0')
