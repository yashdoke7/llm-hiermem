# PyPI Deployment Guide — HierMem

## Prerequisites

```bash
pip install build twine
```

## Step-by-Step First Publish

### 1. Verify Package Config

```bash
# Check pyproject.toml is valid
python -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); print(f'Name: {d[\"project\"][\"name\"]}'); print(f'Version: {d[\"project\"][\"version\"]}')"

# Verify included packages (should NOT include eval/ tests/ results/ docs/)
python -c "
from setuptools import find_packages
pkgs = find_packages(include=['core*','llm*','memory*','retrieval*','baselines*'], exclude=['tests*','eval*','demo*','notebooks*','docs*','results*'])
print('Packages:', pkgs)
"
```

### 2. Build the Package

```bash
python -m build
```

This creates:
- `dist/hiermem-0.1.0.tar.gz` (source distribution)
- `dist/hiermem-0.1.0-py3-none-any.whl` (wheel)

### 3. Verify the Build

```bash
# Check the wheel contents
python -m zipfile -l dist/hiermem-0.1.0-py3-none-any.whl

# Verify metadata
python -m twine check dist/*

# Test install in a fresh venv
python -m venv /tmp/hiermem_test
/tmp/hiermem_test/Scripts/activate      # Windows
pip install dist/hiermem-0.1.0-py3-none-any.whl
python -c "from core.pipeline import HierMemPipeline; print('Import OK')"
python -c "from llm.client import LLMClient; print('LLMClient OK')"
hiermem config
deactivate
```

### 4. Test Publish (TestPyPI)

```bash
# Create account at https://test.pypi.org/account/register/
twine upload --repository testpypi dist/*
# Username: __token__
# Password: your-test-pypi-token

# Test install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ hiermem
```

### 5. Publish to Real PyPI

```bash
# Create account at https://pypi.org/account/register/
# Create API token at https://pypi.org/manage/account/token/

twine upload dist/*
# Username: __token__
# Password: pypi-your-api-token

# Verify
pip install hiermem
python -c "from core.pipeline import HierMemPipeline; print('HierMem installed successfully!')"
```

### 6. Set Up CI/CD (GitHub Actions)

The workflow file is already at `.github/workflows/publish.yml`.

1. Go to GitHub repo → Settings → Secrets and Variables → Actions
2. Add secret: `PYPI_API_TOKEN` = your PyPI API token
3. Create a release on GitHub (tag: `v0.1.0`)
4. CI automatically builds, tests, and publishes

### Version Bumping

For each new release:

1. Update version in `pyproject.toml`:
   ```toml
   version = "0.1.1"
   ```

2. Commit and tag:
   ```bash
   git add pyproject.toml
   git commit -m "Bump version to 0.1.1"
   git tag v0.1.1
   git push origin main --tags
   ```

3. Create a GitHub Release from the tag → CI publishes automatically

## Install Extras

```bash
pip install hiermem            # Core only (pipeline + LLM client)
pip install hiermem[eval]      # + matplotlib, pandas, numpy for benchmarks
pip install hiermem[dev]       # + pytest, ruff, black for development
pip install hiermem[demo]      # + streamlit for interactive demo
pip install hiermem[eval,dev]  # Multiple extras
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `twine check` fails | Ensure README.md is valid markdown |
| Missing packages in wheel | Check `[tool.setuptools.packages.find]` in pyproject.toml |
| Import errors after install | Verify `__init__.py` exists in core/, llm/, memory/, retrieval/ |
| CLI `hiermem` not found | Run `pip install -e .` for development, or reinstall |
