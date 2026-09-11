import unittest

from app.models.age_bins import age_to_bin


class AgeBinContractTest(unittest.TestCase):
    def test_all_boundaries(self):
        cases = {
            0: "0-12", 12: "0-12", 13: "13-19", 19: "13-19",
            20: "20-29", 29: "20-29", 30: "30-39", 39: "30-39",
            40: "40-49", 49: "40-49", 50: "50-64", 64: "50-64",
            65: "65+", 100: "65+",
        }
        for age, expected in cases.items():
            with self.subTest(age=age):
                self.assertEqual(age_to_bin(age), expected)

    def test_continuous_estimates_use_same_bin_contract(self):
        self.assertEqual(age_to_bin(12.9), "0-12")
        self.assertEqual(age_to_bin(19.9), "13-19")
        self.assertEqual(age_to_bin(27.4), "20-29")
        self.assertEqual(age_to_bin(39.9), "30-39")
        self.assertEqual(age_to_bin(49.9), "40-49")
        self.assertEqual(age_to_bin(64.9), "50-64")


if __name__ == "__main__":
    unittest.main()
