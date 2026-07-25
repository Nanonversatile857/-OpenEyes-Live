# 🤝 Contributing to OpenEyes-Live

> First off, thank you for considering contributing to **OpenEyes-Live**!  
> It's contributors like you that make edge AI accessible to everyone.

**Document Version:** v0.1.0
**Last Updated:** 2026-07-25
**Compatible with:** OpenEyes-Live v0.1.x

---

## 📋 Table of Contents

1. [Code of Conduct](#-code-of-conduct)
2. [How You Can Contribute](#-how-you-can-contribute)
3. [Local Development Setup](#-local-development-setup)
4. [Adding a Custom Engine](#-adding-a-custom-engine)
5. [Git Commit Guidelines](#-git-commit-guidelines)
6. [Pull Request Checklist](#-pull-request-checklist)
7. [Issue Reporting Guidelines](#-issue-reporting-guidelines)
8. [Documentation Standards](#-documentation-standards)

---

## 📜 Code of Conduct

This project and everyone participating in it is governed by the [OpenEyes-Live Code of Conduct](./CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [your-email@example.com](mailto:your-email@example.com).

---

## 🗺️ How You Can Contribute

We welcome contributions in all areas, regardless of your experience level.

### For First-Time Contributors (Good First Issues)

| Role | What You Can Do | Difficulty |
| :--- | :--- | :--- |
| 🧪 **Beta Tester** | Test on old phones, file detailed bug reports | 🟢 Easy |
| 📝 **Technical Writer** | Fix typos, improve docs, write tutorials | 🟢 Easy |
| 🐍 **Python Developer** | Fix simple bugs, add CLI improvements | 🟢 Easy |
| 🎨 **UI/UX Designer** | Design terminal UI, suggest improvements | 🟢 Easy |

👉 **[View Good First Issues](https://github.com/vfvincentwong2026/OpenEyes-Live/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)**

### For Experienced Contributors

| Role | What You Can Do | Difficulty |
| :--- | :--- | :--- |
| 🧩 **AI/ML Engineer** | Implement new lightweight vision/audio models | 🟡 Medium |
| ⚡ **Quantization Expert** | `llama.cpp` performance tuning, GGUF conversion | 🟡 Medium |
| 📱 **Mobile Developer** | Termux build scripts, Android/iOS native wrappers | 🟡 Medium |
| 🔌 **MCP Developer** | Adding new tools to the MCP Gateway | 🟡 Medium |
| 🏗️ **Core Developer** | Scheduler, engine manager, core architecture | 🔴 Hard |
| 🔧 **Embedded Engineer** | Old device adaptation, performance optimization | 🔴 Hard |

---

## 🛠️ Local Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub first, then:
git clone https://github.com/YOUR_USERNAME/OpenEyes-Live.git
cd OpenEyes-Live

# Add upstream remote for syncing
git remote add upstream https://github.com/vfvincentwong2026/OpenEyes-Live.git
2. Create a Virtual Environment
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
3. Install Pre-commit Hooks (Optional but Recommended)
bash
# Install pre-commit framework
pip install pre-commit
pre-commit install

# Run on all files (optional)
pre-commit run --all-files
4. Run Tests
bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_scheduler.py
5. Code Style & Linting
bash
# Format code (Black)
black src/ tests/

# Lint code (Flake8)
flake8 src/ tests/

# Type checking (MyPy)
mypy src/

# Sort imports (isort)
isort src/ tests/
6. Sync with Upstream (Before Submitting PR)
bash
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
🧩 Adding a Custom Engine
All engines inherit from BaseEngine located in src/core/engine_manager.py.

Step 1: Create Your Engine File
Create src/engines/my_custom_engine.py:

python
from typing import Dict, Optional
import numpy as np
from src.core.base_engine import BaseEngine
from src.core.types import InferenceResult

class MyCustomEngine(BaseEngine):
    """Custom engine implementation."""

    def __init__(self, model_path: str, config: Optional[Dict] = None):
        super().__init__(model_path)
        self.config = config or {}
        self._loaded = False
        self._model = None

    def load(self) -> None:
        """Load model into memory/VRAM."""
        # Your loading logic here
        self._model = self._load_model(self.model_path)
        self._loaded = True

    def process_frame(self, frame: bytes) -> InferenceResult:
        """Run inference on a single frame."""
        # Your inference logic here
        # frame is raw bytes (JPEG encoded)
        result = self._model.predict(frame)
        return InferenceResult(
            description=result["text"],
            confidence=result["confidence"],
            objects=result.get("objects", []),
            timestamp=datetime.now().isoformat()
        )

    def unload(self) -> None:
        """Free memory."""
        self._model = None
        self._loaded = False
        import gc
        gc.collect()

    @property
    def size_mb(self) -> int:
        return self.config.get("size_mb", 100)

    @property
    def memory_usage_mb(self) -> int:
        if not self._loaded:
            return 0
        # Estimate or measure memory usage
        return self.config.get("memory_mb", 200)

    @property
    def is_loaded(self) -> bool:
        return self._loaded
Step 2: Register Your Engine
Update src/engines/__init__.py:

python
from .base_engine import BaseEngine
from .base_engine import BaseVisionEngine
from .scene_engine import SceneUnderstandingEngine
from .voice_engine import VoiceTranslationEngine
from .memory_engine import MemoryEngine
from .mcp_gateway import MCPGateway
from .my_custom_engine import MyCustomEngine  # Add this

__all__ = [
    "BaseEngine",
    "BaseVisionEngine",
    "SceneUnderstandingEngine",
    "VoiceTranslationEngine",
    "MemoryEngine",
    "MCPGateway",
    "MyCustomEngine",  # Add this
]
Step 3: Add to Registry
Update src/core/registry.yaml:

yaml
engines:
  # ... existing engines ...
  my_custom:
    name: "my_custom_engine"
    model: "MyCustomModel"
    file: "my_custom_model.gguf"
    size_mb: 300
    checksum: "sha256:..."
    source: "https://huggingface.co/your-username/your-model"
Step 4: Write Tests
Create tests/test_my_custom_engine.py:

python
import pytest
from src.engines import MyCustomEngine

def test_engine_load():
    engine = MyCustomEngine("dummy_path")
    assert engine.is_loaded is False
    engine.load()
    assert engine.is_loaded is True

def test_engine_inference():
    engine = MyCustomEngine("dummy_path")
    engine.load()
    result = engine.process_frame(b"fake_frame_data")
    assert result.description is not None
📋 Git Commit Guidelines
We follow the Conventional Commits specification:

Type	Description	Example
feat	A new feature	feat(engine): add SmolVLM2 500M support
fix	A bug fix	fix(cli): resolve camera disconnection crash
docs	Documentation only changes	docs(readme): update quickstart guide
refactor	Code change that neither fixes a bug nor adds a feature	refactor(scheduler): simplify trigger logic
perf	A code change that improves performance	perf(inference): reduce memory usage by 30%
test	Adding missing tests	test(engine): add unit tests for base engine
chore	Changes to build process or auxiliary tools	chore(deps): update llama.cpp to v1.2.0
style	Code style changes (formatting, missing semicolons, etc.)	style(src): apply black formatting
ci	Changes to CI configuration	ci(github): add Python 3.12 to test matrix
Commit Message Format
text
<type>(<scope>): <subject>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
Example:

text
feat(scheduler): add motion detection trigger

Implement frame delta-based motion detection to trigger inference
only when there is meaningful change in the scene.

- Add `motion_threshold` config parameter (default: 0.15)
- Add `min_interval_seconds` to prevent spam
- Add unit tests for trigger logic

Closes #42
🚀 Pull Request Checklist
Before opening a Pull Request (PR), please make sure:

Code Quality
□ Code follows existing formatting guidelines (black, flake8, isort)
□ Type hints are included for all function arguments and returns
□ No print statements (use logging module instead)
□ No commented-out code
□ No TODO comments without issue references
Testing
□ New features include unit tests in /tests
□ Existing tests pass locally (pytest tests/)
□ If adding a new engine, included benchmark data
Documentation
□ README.md is updated if user-facing features changed
□ Docstrings are included for all public functions/classes
□ If adding a new engine, documentation in ENGINES.md is updated
□ CHANGELOG.md is updated with your changes
PR Metadata
□ PR title follows Conventional Commit rules
□ PR description explains what changed and why
□ PR references related issue (e.g., "Fixes #42")
□ PR is not a work-in-progress (WIP) — mark as draft if incomplete
PR Template
When opening a PR, please use the following template:

markdown
## Description
<!-- Describe what this PR does and why it's needed -->

## Type of Change
<!-- Mark with an 'x' -->
- [ ] 🐛 Bug fix
- [ ] ✨ New feature
- [ ] 📝 Documentation update
- [ ] 🏗️ Refactor
- [ ] ⚡ Performance improvement

## Testing
<!-- Describe how you tested this PR -->
- [ ] Tests added/updated
- [ ] Passes `pytest tests/`
- [ ] Passes `flake8 src/`
- [ ] Passes `mypy src/`

## Related Issues
<!-- Link related issues here -->
Closes #(issue_number)

## Screenshots (if applicable)
<!-- Add screenshots to help explain your changes -->

## Additional Notes
<!-- Any additional information for reviewers -->
🐛 Issue Reporting Guidelines
Before Filing an Issue
Search existing issues (open and closed) to avoid duplicates

Check the Troubleshooting section

Test with the latest version (main branch)

Bug Report Template
markdown
## Description
<!-- A clear and concise description of the bug -->

## Steps to Reproduce
1. Run `openeyes watch --source camera --engines base`
2. ...

## Expected Behavior
<!-- What should have happened -->

## Actual Behavior
<!-- What actually happened -->

## Environment
- Device: [e.g., Xiaomi 6]
- OS: [e.g., Android 9]
- Python Version: [e.g., 3.10.12]
- OpenEyes-Live Version: [e.g., v0.1.0]

## Logs
Paste relevant logs here

text

## Screenshots (if applicable)
<!-- Add screenshots to help explain your problem -->
Feature Request Template
markdown
## Problem Statement
<!-- What problem does this feature solve? -->

## Proposed Solution
<!-- How should this feature work? -->

## Alternatives Considered
<!-- What alternatives did you consider? -->

## Additional Context
<!-- Any other information -->
📚 Documentation Standards
Adding or Updating Documentation
Document Type	Location	Purpose
Project Overview	README.md	First impression, quick start
Detailed Architecture	docs/ARCHITECTURE.md	Technical deep dive
Performance Data	docs/PERFORMANCE_BENCHMARK.md	Benchmarks and compatibility
Mobile Setup	docs/QUICKSTART_MOBILE.md	Step-by-step mobile guide
API Reference	docs/API_REFERENCE.md	Python API documentation
Contributing	CONTRIBUTING.md	This document
Code of Conduct	CODE_OF_CONDUCT.md	Community standards
Documentation Style
Element	Style
Headers	Title Case for main headers
Code Blocks	Use triple backticks with language specifier
Links	Use descriptive link text, not "click here"
Lists	Use - for unordered lists, 1. for ordered
Emphasis	Use **bold** for emphasis, *italic* for subtle emphasis
Commands	Use backticks for inline commands: `openeyes watch`
File Paths	Use backticks: `src/core/scheduler.py`
📄 License
By contributing to OpenEyes-Live, you agree that your contributions will be licensed under the project's Apache License 2.0.

💬 Getting Help
Channel	Purpose
GitHub Issues	Bug reports, feature requests
GitHub Discussions	General questions, ideas, announcements
Discord (future)	Real-time community chat
Document Version: v0.1.0
Last Updated: 2026-07-25
Compatible with: OpenEyes-Live v0.1.x