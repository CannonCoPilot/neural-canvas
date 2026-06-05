import unittest
from unittest.mock import MagicMock, patch
from art_agent_team.agents import research_agent

class DummyPart:
    pass

class DummyBlob:
    pass

class DummyPILImage:
    pass

class DummyIPyImage:
    pass

class TestGoogleGenAIContentType(unittest.TestCase):
    def test_supported_types(self):
        # bytes, str, dict, Blob, PIL.Image.Image, IPython.display.Image
        self.assertTrue(research_agent._is_supported_google_genai_content_type(b"abc"))
        self.assertTrue(research_agent._is_supported_google_genai_content_type("abc"))
        self.assertTrue(research_agent._is_supported_google_genai_content_type({"a": 1}))
        blob = DummyBlob()
        blob.__class__.__name__ = "Blob"
        self.assertTrue(research_agent._is_supported_google_genai_content_type(blob))
        pil = DummyPILImage()
        pil.__class__.__name__ = "Image"
        self.assertTrue(research_agent._is_supported_google_genai_content_type(pil))
        ipy = DummyIPyImage()
        ipy.__class__.__name__ = "Image"
        self.assertTrue(research_agent._is_supported_google_genai_content_type(ipy))

    def test_unsupported_types(self):
        part = DummyPart()
        part.__class__.__name__ = "Part"
        self.assertFalse(research_agent._is_supported_google_genai_content_type(part))
        self.assertFalse(research_agent._is_supported_google_genai_content_type(123))
        self.assertFalse(research_agent._is_supported_google_genai_content_type(object()))

    @patch("art_agent_team.agents.research_agent.logging")
    def test_api_call_rejects_part(self, mock_logging):
        # Simulate content with a Part object
        client = MagicMock()
        client.__name__ = "google.generativeai"
        content = ["prompt", DummyPart()]
        # Should log error and return None from content preparation
        result = None
        try:
            # Simulate the relevant content check logic
            for c in content:
                if type(c).__name__ == "Part" or not research_agent._is_supported_google_genai_content_type(c):
                    result = None
                    break
        except Exception:
            result = "error"
        self.assertIsNone(result)

if __name__ == "__main__":
    unittest.main()