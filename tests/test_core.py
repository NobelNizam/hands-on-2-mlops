import unittest

from src.config import MIN_PRECISION, PROMOTION_MARGIN
from src.data import dataset_md5, expected_split
from src.register import passes_quality_gate


class DataTest(unittest.TestCase):
    def test_split_is_reproducible_and_stratified(self):
        first_train, first_test = expected_split()
        second_train, second_test = expected_split()
        self.assertTrue(first_train.equals(second_train))
        self.assertTrue(first_test.equals(second_test))
        self.assertEqual((len(first_train), len(first_test)), (455, 114))
        self.assertAlmostEqual(first_train["target"].mean(), first_test["target"].mean(), delta=0.01)

    def test_hash_is_stable(self):
        self.assertEqual(dataset_md5(), dataset_md5())
        self.assertEqual(len(dataset_md5()), 32)


class QualityGateTest(unittest.TestCase):
    def test_accepts_sufficient_improvement(self):
        self.assertTrue(
            passes_quality_gate(MIN_PRECISION, 0.90 + PROMOTION_MARGIN, 0.90)
        )

    def test_rejects_low_precision_or_small_improvement(self):
        self.assertFalse(passes_quality_gate(MIN_PRECISION - 0.01, 1.0, 0.90))
        self.assertFalse(passes_quality_gate(1.0, 0.90, 0.90))


if __name__ == "__main__":
    unittest.main()
