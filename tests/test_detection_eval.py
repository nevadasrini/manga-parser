"""Tests for greedy IoU eval helpers (panels / bubbles shared math)."""

from __future__ import annotations

import unittest

from manga109_loader import BBox, Manga109Annotation, Manga109Page

from panel_eval_lib import gt_text_boxes_xyxy, match_greedy, xyxy_iou


class TestGreedy(unittest.TestCase):
    def test_xyxy_iou_overlap(self):
        self.assertAlmostEqual(xyxy_iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertAlmostEqual(xyxy_iou((0, 0, 10, 10), (10, 10, 20, 20)), 0.0)

    def test_match_one_to_one(self):
        gts = [(0.0, 0.0, 10.0, 10.0)]
        preds = [(0.0, 0.0, 10.0, 10.0)]
        tp, fp, fn = match_greedy(gts, preds, 0.5)
        self.assertEqual((tp, fp, fn), (1, 0, 0))

    def test_match_fp_fn(self):
        gts = [(0.0, 0.0, 10.0, 10.0)]
        preds = [(100.0, 100.0, 110.0, 110.0)]  # no overlap → FP + FN (not greedy match)
        tp, fp, fn = match_greedy(gts, preds, 0.5)
        self.assertEqual((tp, fp, fn), (0, 1, 1))

    def test_gt_text_boxes(self):
        page = Manga109Page(
            book_title="Test",
            page_index=1,
            image_path="/tmp/a.jpg",
            width=800,
            height=1200,
            annotations=[
                Manga109Annotation(
                    ann_id="1",
                    category="text",
                    bbox=BBox(10, 20, 110, 120),
                )
            ],
        )
        bx = gt_text_boxes_xyxy(page)
        self.assertEqual(len(bx), 1)
        self.assertEqual(bx[0], (10.0, 20.0, 110.0, 120.0))


if __name__ == "__main__":
    unittest.main()
