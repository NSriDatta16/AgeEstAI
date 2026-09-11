import ast
import unittest
from pathlib import Path


class AgeBinContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = (Path(__file__).parents[1] / "app" / "models" / "infer.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "age_to_bin")
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "infer.py", "exec"), namespace)
        cls.age_to_bin = staticmethod(namespace["age_to_bin"])

    def test_all_boundaries(self):
        cases = {
            0: "0-12", 12: "0-12", 13: "13-19", 19: "13-19",
            20: "20-29", 29: "20-29", 30: "30-39", 39: "30-39",
            40: "40-49", 49: "40-49", 50: "50-64", 64: "50-64",
            65: "65+", 100: "65+",
        }
        for age, expected in cases.items():
            with self.subTest(age=age):
                self.assertEqual(self.age_to_bin(age), expected)

    def test_continuous_estimates_use_same_bin_contract(self):
        self.assertEqual(self.age_to_bin(27.4), "20-29")
        self.assertEqual(self.age_to_bin(49.9), "40-49")
        self.assertEqual(self.age_to_bin(64.9), "65+")


if __name__ == "__main__":
    unittest.main()
