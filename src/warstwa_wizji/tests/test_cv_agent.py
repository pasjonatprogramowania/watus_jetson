"""Testy inicjalizacji CVAgent z mockowanymi komponentami."""

import unittest
from unittest.mock import MagicMock, patch


class TestCVAgentInit(unittest.TestCase):

    @patch("warstwa_wizji.src.cv_agent.DetectionPipeline")
    @patch("warstwa_wizji.src.cv_agent.PersonAnalyzer")
    @patch("warstwa_wizji.src.cv_agent.VideoIO")
    @patch("warstwa_wizji.src.cv_agent.ModelManager")
    def test_creates_all_components(self, MockMM, MockVIO, MockPA, MockDP):
        from warstwa_wizji import CVAgent
        agent = CVAgent(weights_path="fake.pt", source=0)
        MockMM.assert_called_once()
        MockVIO.assert_called_once()
        MockPA.assert_called_once()
        MockDP.assert_called_once()
        self.assertIsNotNone(agent.models)
        self.assertIsNotNone(agent.video)
        self.assertIsNotNone(agent.person_analyzer)
        self.assertIsNotNone(agent.pipeline)

    @patch("warstwa_wizji.src.cv_agent.DetectionPipeline")
    @patch("warstwa_wizji.src.cv_agent.PersonAnalyzer")
    @patch("warstwa_wizji.src.cv_agent.VideoIO")
    @patch("warstwa_wizji.src.cv_agent.ModelManager")
    def test_json_save_func_stored(self, MockMM, MockVIO, MockPA, MockDP):
        from warstwa_wizji import CVAgent
        save_fn = MagicMock()
        agent = CVAgent(weights_path="fake.pt", json_save_func=save_fn)
        self.assertIs(agent.save_to_json, save_fn)

    @patch("warstwa_wizji.src.cv_agent.DetectionPipeline")
    @patch("warstwa_wizji.src.cv_agent.PersonAnalyzer")
    @patch("warstwa_wizji.src.cv_agent.VideoIO")
    @patch("warstwa_wizji.src.cv_agent.ModelManager")
    def test_export_engine_flag_passed(self, MockMM, MockVIO, MockPA, MockDP):
        from warstwa_wizji import CVAgent
        agent = CVAgent(weights_path="fake.pt", export_to_engine=True)
        call_kwargs = MockMM.call_args
        self.assertTrue(call_kwargs.kwargs.get("export_to_engine", False)
                        or (len(call_kwargs.args) > 2 and call_kwargs.args[2]))
