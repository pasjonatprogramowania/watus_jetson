"""Testy obliczeń kąta obiektu względem osi kamery."""

import math
import unittest


class TestCalcObjAngle(unittest.TestCase):

    def setUp(self):
        from warstwa_wizji import calc_obj_angle
        self.calc = calc_obj_angle
        self.IMAGE_W = 640

    def test_center_object_angle_near_zero(self):
        angle = self.calc((300, 100), (340, 200), self.IMAGE_W, fov_deg=102)
        self.assertAlmostEqual(angle, 0.0, delta=2.0)

    def test_right_object_positive_angle(self):
        angle = self.calc((500, 100), (600, 200), self.IMAGE_W, fov_deg=102)
        self.assertGreater(angle, 0)

    def test_left_object_negative_angle(self):
        angle = self.calc((40, 100), (140, 200), self.IMAGE_W, fov_deg=102)
        self.assertLess(angle, 0)

    def test_symmetry(self):
        offset = 200
        center = self.IMAGE_W / 2
        box_w = 50
        right = self.calc(
            (center + offset, 100), (center + offset + box_w, 200),
            self.IMAGE_W, fov_deg=102,
        )
        left = self.calc(
            (center - offset - box_w, 100), (center - offset, 200),
            self.IMAGE_W, fov_deg=102,
        )
        self.assertAlmostEqual(right, -left, delta=0.5)

    def test_different_fov(self):
        narrow = self.calc((500, 0), (550, 50), self.IMAGE_W, fov_deg=60)
        wide = self.calc((500, 0), (550, 50), self.IMAGE_W, fov_deg=120)
        # Szerszy FOV → mniejszy kąt dla tego samego piksela
        self.assertLess(abs(narrow), abs(wide))

    def test_edge_object_returns_finite(self):
        angle = self.calc((0, 0), (10, 10), self.IMAGE_W, fov_deg=102)
        self.assertTrue(math.isfinite(angle))
