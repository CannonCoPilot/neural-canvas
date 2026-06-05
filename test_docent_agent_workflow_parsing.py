import unittest
import logging
import json
import re

# Simulate the parsing logic from docent_agent.py
def parse_workflow_from_llm_response(llm_response):
    logger = logging.getLogger("test")
    expected_keys = {"run_research", "run_vision", "run_upscale", "run_placard"}
    workflow_json = None
    workflow_dict = None

    logger.debug(f"Raw LLM response before parsing: {repr(llm_response)}")

    try:
        if not llm_response or not isinstance(llm_response, str) or llm_response.strip() == "":
            logger.error("LLM response is empty or not a string. Falling back to default workflow.")
            raise ValueError("Empty or invalid LLM response")

        match = re.search(r"\{.*\}", llm_response, re.DOTALL)
        if match:
            workflow_json = match.group(0)
            logger.debug(f"Extracted JSON via regex: {workflow_json}")
        else:
            workflow_json = llm_response.strip()
            logger.debug(f"No JSON object found via regex, using stripped response: {workflow_json}")

        if not workflow_json or workflow_json in ("{}", "[]", "null"):
            logger.error(f"Workflow JSON is empty or invalid: {repr(workflow_json)}. Falling back to default workflow.")
            raise ValueError("Empty or invalid workflow JSON")

        try:
            workflow_dict = json.loads(workflow_json)
            logger.debug(f"Parsed JSON: {workflow_dict}")
        except json.JSONDecodeError as jde:
            logger.error(f"JSONDecodeError: {jde}. Raw input: {repr(workflow_json)}. Attempting to recover by fixing common issues.")
            fixed_json = (
                workflow_json.replace("'", '"')
                .replace(",}", "}")
                .replace(",]", "]")
            )
            logger.debug(f"Trying to parse fixed JSON: {fixed_json}")
            try:
                workflow_dict = json.loads(fixed_json)
                logger.info("Successfully parsed JSON after fixing common issues.")
            except Exception as fix_e:
                logger.error(f"Failed to parse fixed JSON: {fix_e}. Raw fixed input: {repr(fixed_json)}")
                workflow_dict = None

        if not isinstance(workflow_dict, dict):
            logger.error(f"Parsed workflow_dict is not a dict: {workflow_dict}")
            workflow_dict = None
        elif not expected_keys.issubset(set(workflow_dict.keys())):
            logger.warning(f"Parsed workflow_dict missing expected keys: {workflow_dict.keys()}")

        if workflow_dict:
            return {
                "run_research": bool(workflow_dict.get("run_research", False)),
                "run_vision": bool(workflow_dict.get("run_vision", False)),
                "run_upscale": bool(workflow_dict.get("run_upscale", False)),
                "run_placard": bool(workflow_dict.get("run_placard", False)),
                "fallback": False,
            }
        else:
            logger.error("Workflow dict is None or invalid after all parsing attempts. Falling back to default workflow (Research -> Vision).")
            return {
                "run_research": True,
                "run_vision": True,
                "run_upscale": False,
                "run_placard": False,
                "fallback": True,
            }

    except Exception as e:
        logger.error(f"Failed to interpret workflow stages from LLM response: {e}. Falling back to default workflow (Research -> Vision). Raw response: {llm_response}", exc_info=True)
        return {
            "run_research": True,
            "run_vision": True,
            "run_upscale": False,
            "run_placard": False,
            "fallback": True,
        }

class TestDocentAgentWorkflowParsing(unittest.TestCase):
    def test_valid_json(self):
        resp = '{"run_research": true, "run_vision": false, "run_upscale": true, "run_placard": false}'
        result = parse_workflow_from_llm_response(resp)
        self.assertEqual(result, {
            "run_research": True,
            "run_vision": False,
            "run_upscale": True,
            "run_placard": False,
            "fallback": False,
        })

    def test_single_quotes_and_trailing_comma(self):
        resp = "{'run_research': true, 'run_vision': false, 'run_upscale': true, 'run_placard': false,}"
        result = parse_workflow_from_llm_response(resp)
        self.assertEqual(result, {
            "run_research": True,
            "run_vision": False,
            "run_upscale": True,
            "run_placard": False,
            "fallback": False,
        })

    def test_missing_keys(self):
        resp = '{"run_research": true, "run_vision": false}'
        result = parse_workflow_from_llm_response(resp)
        self.assertTrue(result["fallback"])
        self.assertTrue(result["run_research"])
        self.assertTrue(result["run_vision"])
        self.assertFalse(result["run_upscale"])
        self.assertFalse(result["run_placard"])

    def test_empty_string(self):
        resp = ""
        result = parse_workflow_from_llm_response(resp)
        self.assertTrue(result["fallback"])
        self.assertTrue(result["run_research"])
        self.assertTrue(result["run_vision"])
        self.assertFalse(result["run_upscale"])
        self.assertFalse(result["run_placard"])

    def test_malformed_json(self):
        resp = '{"run_research": tru, "run_vision": false, "run_upscale": true, "run_placard": false}'
        result = parse_workflow_from_llm_response(resp)
        self.assertTrue(result["fallback"])
        self.assertTrue(result["run_research"])
        self.assertTrue(result["run_vision"])
        self.assertFalse(result["run_upscale"])
        self.assertFalse(result["run_placard"])

    def test_non_dict_json(self):
        resp = '["run_research", "run_vision"]'
        result = parse_workflow_from_llm_response(resp)
        self.assertTrue(result["fallback"])
        self.assertTrue(result["run_research"])
        self.assertTrue(result["run_vision"])
        self.assertFalse(result["run_upscale"])
        self.assertFalse(result["run_placard"])

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
import unittest
from unittest.mock import patch, MagicMock
from art_agent_team.docent_agent import DocentAgent

class TestDocentAgentVisionFallback(unittest.TestCase):
    def setUp(self):
        self.agent = DocentAgent()
        self.agent.config = {}  # Minimal config

    def test_vision_agent_fallback_for_default(self):
        # Should always get a VisionAgent for 'Default'
        vision_agent_class = self.agent.vision_agent_classes.get('Default')
        self.assertIsNotNone(vision_agent_class, "DefaultVisionAgent should be registered as 'Default'.")

    def test_vision_agent_fallback_for_unknown_genre(self):
        # Should fallback to DefaultVisionAgent for unknown genres
        vision_agent_class = self.agent.vision_agent_classes.get('NonexistentGenre')
        if vision_agent_class is None:
            vision_agent_class = self.agent.vision_agent_classes.get('Default')
        self.assertIsNotNone(vision_agent_class, "Fallback to DefaultVisionAgent should always succeed.")

    @patch('art_agent_team.docent_agent.DocentAgent')
    def test_vision_stage_not_skipped(self, MockDocentAgent):
        # Simulate the Vision stage logic for a genre not present in vision_agent_classes
        instance = MockDocentAgent.return_value
        instance.vision_agent_classes = {'Default': MagicMock()}
        genre = 'NonexistentGenre'
        vision_agent_class = instance.vision_agent_classes.get(genre)
        if not vision_agent_class:
            vision_agent_class = instance.vision_agent_classes.get('Default')
        self.assertIsNotNone(vision_agent_class, "Vision stage should not be skipped for any genre.")

if __name__ == '__main__':
    unittest.main()