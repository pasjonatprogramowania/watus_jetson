"""Testy klasy ParallelProcess."""

import multiprocessing
import unittest


class TestParallelProcess(unittest.TestCase):

    def test_creation(self):
        from warstwa_wizji.src.utils.parallel import ParallelProcess
        proc = ParallelProcess(process_id=42)
        self.assertEqual(proc.process_id, 42)

    def test_alias(self):
        from warstwa_wizji.src.utils.parallel import ParallelProcess, Process
        self.assertIs(Process, ParallelProcess)

    def test_is_multiprocessing_process(self):
        from warstwa_wizji.src.utils.parallel import ParallelProcess
        proc = ParallelProcess(process_id=1)
        self.assertIsInstance(proc, multiprocessing.Process)
