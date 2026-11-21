from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from perception_service.config import load_local_environment
from perception_service.semantics import SemanticSettings


class ConfigTests(unittest.TestCase):
    def test_loads_local_values_without_overwriting_process_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env.local"
            path.write_text(
                "DEEPSEEK_API_KEY=from-file\n"
                "DEEPSEEK_MODEL=deepseek-flash\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"DEEPSEEK_API_KEY": "from-process"},
                clear=False,
            ):
                loaded = load_local_environment(path)
                self.assertEqual(loaded["DEEPSEEK_API_KEY"], "from-file")
                self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "from-process")

    def test_selects_deepseek_flash(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SEMANTIC_PROVIDER": "deepseek",
                "DEEPSEEK_API_KEY": "test-key",
                "DEEPSEEK_MODEL": "deepseek-flash",
                "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
            },
            clear=True,
        ):
            settings = SemanticSettings.from_environment()
        self.assertEqual(settings.provider, "deepseek")
        self.assertEqual(settings.model, "deepseek-flash")
        self.assertEqual(settings.base_url, "https://api.deepseek.com")


if __name__ == "__main__":
    unittest.main()
