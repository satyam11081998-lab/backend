"""
Unit and Integration Test Suite for the Automated Deck Ingestion Pipeline.
"""

import os
import sys
import tempfile
import unittest

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.deck_classifier import classify_deck_rules
from services.deck_extractor import (
    compute_file_hash,
    extract_deck,
    extract_numbers_from_text,
    mine_path_and_filename_signals,
    normalize_extracted_text,
)
from services.deck_taxonomy import (
    CASE_TYPES,
    INDUSTRIES,
    KNOWN_COMPETITIONS,
    RESULTS,
    ROUND_TYPES,
    SOURCE_KINDS,
)
from services.gdrive import (
    normalize_deck_filename,
    sanitize_filename_component,
)


class TestDeckIngestionPipeline(unittest.TestCase):

    def test_sha256_hash_and_deduplication(self):
        """Verify that file hashing is deterministic and collision-safe."""
        data_a = b"%PDF-1.4 Mock PDF content for case challenge"
        data_b = b"%PDF-1.4 Mock PDF content for case challenge"
        data_c = b"%PDF-1.4 Different content"

        hash_a = compute_file_hash(data_a)
        hash_b = compute_file_hash(data_b)
        hash_c = compute_file_hash(data_c)

        self.assertEqual(len(hash_a), 64)
        self.assertEqual(hash_a, hash_b)
        self.assertNotEqual(hash_a, hash_c)

    def test_path_signal_mining_accenture(self):
        """Verify path signal extraction on real Accenture test path."""
        sample_path = r"Corporate Case Comps/Accenture B-School Challenge- National Finalist/Aviators_Pranjal Tyagi.pdf"
        signals = mine_path_and_filename_signals(sample_path)

        self.assertEqual(signals["detected_competition"], "Accenture B-School Challenge")
        self.assertEqual(signals["detected_company"], "Accenture")
        self.assertEqual(signals["detected_result"], "National Finalist")
        self.assertEqual(signals["source_kind"], "corporate")

    def test_path_signal_mining_colgate(self):
        """Verify path signal extraction on Colgate Transcend path."""
        sample_path = r"Corporate Case Comps/Colgate Transcend National Runner Up/team_alpha.pdf"
        signals = mine_path_and_filename_signals(sample_path)

        self.assertEqual(signals["detected_competition"], "Colgate Transcend")
        self.assertEqual(signals["detected_company"], "Colgate-Palmolive")
        self.assertEqual(signals["detected_result"], "National 1st Runner Up")

    def test_path_signal_mining_hul_lime(self):
        """Verify path signal extraction on HUL LIME Winners."""
        sample_path = r"Corporate Case Comps/HUL LIME National Winners/deck_v1.pdf"
        signals = mine_path_and_filename_signals(sample_path)

        self.assertEqual(signals["detected_competition"], "HUL L.I.M.E.")
        self.assertEqual(signals["detected_company"], "Hindustan Unilever")
        self.assertEqual(signals["detected_result"], "National Winner")

    def test_path_signal_mining_bschool_filename(self):
        """Verify filename parsing with B-school, team, company and finance clues."""
        sample_path = r"Corporate Case Comps/Multiple Corporate Decks/MDIGurgaon_Helios_Cummins_Fin.pdf"
        signals = mine_path_and_filename_signals(sample_path)

        self.assertEqual(signals["detected_college"], "MDI Gurgaon")
        self.assertEqual(signals["detected_company"], "Cummins")

    def test_rule_classification_conforms_to_taxonomy(self):
        """Verify rule classifier outputs strictly controlled taxonomy vocabulary."""
        mock_extracted = {
            "file_hash": "mockhash123",
            "file_size": 1024,
            "file_type": "pdf",
            "slide_count": 12,
            "full_text": "Supply chain optimization and reverse logistics for FMCG distribution network.",
            "slides": [{"slide_number": 1, "text": "Supply Chain Strategy for FMCG", "title": "Supply Chain"}],
            "numbers": {"2024", "15"},
            "path_signals": {
                "detected_competition": "Mondelez Supply Track",
                "detected_company": "Mondelez International",
                "detected_result": "National Finalist",
                "detected_round_type": "finale",
                "detected_year": 2024,
                "source_kind": "corporate",
            },
        }

        classification = classify_deck_rules(mock_extracted)

        self.assertIn(classification["case_type"], CASE_TYPES)
        self.assertIn(classification["result"], RESULTS)
        self.assertIn(classification["round_type"], ROUND_TYPES)
        self.assertIn(classification["source_kind"], SOURCE_KINDS)
        self.assertIn(classification["industry"], INDUSTRIES)
        self.assertGreaterEqual(classification["confidence"], 0.70)
        self.assertIsInstance(classification["field_confidences"], dict)

    def test_filename_normalization_and_sanitization(self):
        """Verify deterministic sanitized filename generator."""
        normalized = normalize_deck_filename(
            competition="Asian Paints Canvas 2025: Challenge?",
            company="Asian Paints Ltd.",
            case_type="growth strategy",
            year=2025,
            ext="pdf",
        )

        self.assertEqual(normalized, "Asian_Paints_Canvas_2025_Challenge_Growth_Strategy_2025.pdf")
        # Assert no illegal characters
        for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
            self.assertNotIn(char, normalized)

    def test_number_verification_and_zero_hallucination(self):
        """Verify number extraction logic handles formatted numbers, percentages, and decimals."""
        text = "Revenue grew by 24.5% to reach Rs 1,500 crore in 2024 with 3 distinct pillars."
        numbers = extract_numbers_from_text(text)

        self.assertIn("24.5", numbers)
        self.assertIn("1500", numbers)
        self.assertIn("2024", numbers)
        self.assertIn("3", numbers)
        # Should not contain hallucinated 99 or 5000
        self.assertNotIn("99", numbers)


if __name__ == "__main__":
    unittest.main()
