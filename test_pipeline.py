import unittest

from model_pipeline import TARGET, load_data, prepare_features, train_model


class TestOnlineFoodPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = load_data()

    def test_dataset_not_empty(self):
        self.assertGreater(len(self.df), 100)

    def test_target_column_exists(self):
        self.assertIn(TARGET, self.df.columns)

    def test_target_has_only_yes_no(self):
        values = set(self.df[TARGET].str.strip().str.capitalize().unique())
        self.assertTrue(values.issubset({"Yes", "No"}))

    def test_no_missing_target(self):
        self.assertEqual(self.df[TARGET].isnull().sum(), 0)

    def test_features_match_rows(self):
        X, y = prepare_features(self.df)
        self.assertEqual(len(X), len(y))

    def test_model_accuracy_above_threshold(self):
        _, accuracy = train_model(self.df)
        self.assertGreaterEqual(accuracy, 0.70)


if __name__ == "__main__":
    unittest.main()
