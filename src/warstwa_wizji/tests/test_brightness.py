"""Testy obliczania jasności i sugestii trybu wyświetlania."""

import unittest
import numpy as np


class TestBrightness(unittest.TestCase):

    def setUp(self):
        from warstwa_wizji import calc_brightness, suggest_mode
        self.calc_brightness = calc_brightness
        self.suggest_mode = suggest_mode

    def test_black_image_brightness_near_zero(self):
        black = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertAlmostEqual(self.calc_brightness(black), 0.0, delta=0.01)

    def test_white_image_brightness_near_one(self):
        white = np.full((100, 100, 3), 255, dtype=np.uint8)
        self.assertAlmostEqual(self.calc_brightness(white), 1.0, delta=0.01)

    def test_gray_image_brightness_about_half(self):
        gray = np.full((100, 100, 3), 128, dtype=np.uint8)
        val = self.calc_brightness(gray)
        self.assertGreater(val, 0.4)
        self.assertLess(val, 0.6)

    def test_brightness_range(self):
        random_img = np.random.randint(0, 256, (200, 300, 3), dtype=np.uint8)
        val = self.calc_brightness(random_img)
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)

    def test_large_image_works(self):
        big = np.full((1920, 1080, 3), 100, dtype=np.uint8)
        val = self.calc_brightness(big)
        self.assertGreater(val, 0.0)

    def test_suggest_mode_dark_to_light(self):
        mode = self.suggest_mode(0.8, "dark")
        self.assertEqual(mode, "light")

    def test_suggest_mode_light_to_dark(self):
        mode = self.suggest_mode(0.2, "light")
        self.assertEqual(mode, "dark")

    def test_suggest_mode_hysteresis_no_change(self):
        mode = self.suggest_mode(0.5, "light")
        self.assertEqual(mode, "light")
        mode = self.suggest_mode(0.5, "dark")
        self.assertEqual(mode, "dark")
