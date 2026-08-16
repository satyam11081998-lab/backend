import unittest
from services.deck_ai import _numbers_in_text

class TestDeckAIValidation(unittest.TestCase):
    def test_number_extraction_and_hallucination_detection(self):
        source_text = "The market size is 500 million and target revenue growth is 25% by 2026."
        source_numbers = _numbers_in_text(source_text)
        self.assertEqual(source_numbers, {"500", "25", "2026"})

        # Summary with only allowed numbers
        good_summary = "In 2026, the company captured 25% of the 500 million market."
        good_numbers = _numbers_in_text(good_summary)
        invented = good_numbers - source_numbers
        self.assertEqual(invented, set())

        # Summary with invented numbers (e.g. 34% or 100)
        bad_summary = "In 2026, the company captured 34% of the 500 million market and added 100 clients."
        bad_numbers = _numbers_in_text(bad_summary)
        invented = bad_numbers - source_numbers
        self.assertEqual(invented, {"34", "100"})

if __name__ == "__main__":
    unittest.main()
