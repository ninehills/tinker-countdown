import unittest

from eval import check_format_tags


class CheckFormatTagsTest(unittest.TestCase):
    def test_valid_with_think_and_answer(self):
        sample = "<think>reasoning</think> something <answer>1 + 2 = 3</answer>"
        self.assertTrue(check_format_tags(sample))

    def test_valid_with_newlines(self):
        sample = "<think>\nstep1\n</think>\n<answer>\n4 * (2 + 1)\n</answer>"
        self.assertTrue(check_format_tags(sample))

    def test_missing_think(self):
        sample = "No think tag <answer>1+1=2</answer>"
        self.assertFalse(check_format_tags(sample))

    def test_missing_answer(self):
        sample = "<think>reasoning only</think>"
        self.assertFalse(check_format_tags(sample))

    def test_duplicate_think(self):
        sample = "<think>a</think> mid <think>b</think> <answer>1</answer>"
        self.assertFalse(check_format_tags(sample))

    def test_duplicate_answer(self):
        sample = "<think>a</think> <answer>1</answer> extra <answer>2</answer>"
        self.assertFalse(check_format_tags(sample))

    def test_answer_before_think(self):
        sample = "<answer>1</answer> text <think>later</think>"
        self.assertFalse(check_format_tags(sample))


if __name__ == "__main__":
    unittest.main()
