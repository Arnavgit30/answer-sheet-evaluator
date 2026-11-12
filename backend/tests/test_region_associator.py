"""Unit tests for region_associator."""

import unittest

from backend.app.region_associator import associate_regions


class RegionAssociatorTests(unittest.TestCase):
    def test_multiple_questions_on_one_page(self):
        pages = [
            {
                "page_number": 1,
                "elements": [
                    {"type": "text", "text": "Q1 Explain photosynthesis."},
                    {"type": "text", "text": "It converts light energy."},
                    {"type": "text", "text": "Q2 What is evaporation?"},
                    {"type": "text", "text": "Liquid becomes vapour."},
                ],
            }
        ]
        q_map, unassigned = associate_regions(pages)
        self.assertEqual(list(q_map.keys()), ["Q1", "Q2"])
        self.assertEqual(len(unassigned), 0)
        self.assertEqual(len(q_map["Q1"]), 2)
        self.assertEqual(len(q_map["Q2"]), 2)
        # Verify region IDs are unique and disjoint
        r_q1 = {ar.region.region_id for ar in q_map["Q1"]}
        r_q2 = {ar.region.region_id for ar in q_map["Q2"]}
        self.assertEqual(r_q1 & r_q2, set())

    def test_one_question_spanning_multiple_pages(self):
        pages = [
            {
                "page_number": 1,
                "elements": [
                    {"type": "text", "text": "Q1 First page discussion on distributed systems."}
                ],
            },
            {
                "page_number": 2,
                "elements": [
                    {"type": "text", "text": "Continuing Q1 on second page with cluster architecture."}
                ],
            },
            {
                "page_number": 3,
                "elements": [
                    {"type": "text", "text": "Final part of Q1 answer."}
                ],
            },
        ]
        q_map, unassigned = associate_regions(pages)
        self.assertEqual(list(q_map.keys()), ["Q1"])
        self.assertEqual(len(q_map["Q1"]), 3)
        self.assertEqual([ar.region.page_number for ar in q_map["Q1"]], [1, 2, 3])

    def test_q1_ending_and_q2_starting_on_same_page(self):
        pages = [
            {
                "page_number": 1,
                "elements": [
                    {"type": "text", "text": "Q1 First question started here."},
                ],
            },
            {
                "page_number": 2,
                "elements": [
                    {"type": "text", "text": "Q1 finished on top of page 2."},
                    {"type": "text", "text": "Q2 started below on page 2."},
                ],
            },
        ]
        q_map, _ = associate_regions(pages)
        self.assertEqual(len(q_map["Q1"]), 2)
        self.assertEqual(len(q_map["Q2"]), 1)
        self.assertEqual(q_map["Q1"][1].region.text, "Q1 finished on top of page 2.")
        self.assertEqual(q_map["Q2"][0].region.text, "Q2 started below on page 2.")

    def test_subparts_stay_inside_parent_question(self):
        pages = [
            {
                "page_number": 1,
                "elements": [
                    {"type": "text", "text": "Q3 Explain the following:"},
                    {"type": "text", "text": "(a) Acid properties"},
                    {"type": "text", "text": "Turns blue litmus red."},
                    {"type": "text", "text": "(b) Base properties"},
                    {"type": "text", "text": "Turns red litmus blue."},
                ],
            }
        ]
        q_map, _ = associate_regions(pages)
        self.assertEqual(list(q_map.keys()), ["Q3"])
        self.assertEqual(len(q_map["Q3"]), 5)

    def test_numbered_list_inside_answer_does_not_split_question(self):
        pages = [
            {
                "page_number": 1,
                "elements": [
                    {"type": "text", "text": "Q1 Describe cluster computing components."},
                    {"type": "text", "text": "Key components are:"},
                    {"type": "text", "text": "1. Master node"},
                    {"type": "text", "text": "1. Backup node"},
                ],
            }
        ]
        q_map, _ = associate_regions(pages)
        self.assertEqual(list(q_map.keys()), ["Q1"])
        self.assertEqual(len(q_map["Q1"]), 4)


if __name__ == "__main__":
    unittest.main()
