# Troubleshooting

This page contains solutions to common problems you might encounter when using NanofinderParser.

## Installation Problems

### Package not found

If you get a "package not found" error:

1. Check that you're using the correct package name: `nanofinderparser`
2. Ensure you have Python 3.12+ installed
3. Try upgrading pip: `pip install --upgrade pip`

## Runtime Issues

### Import errors

If you get import errors:

```python
import nanofinderparser  # Correct import
```

Make sure you're using the correct import name: `nanofinderparser`.

## CLI Issues

### Command not found

If the `nanofinderparser` command is not found:

1. Ensure the package is installed: `pip list | grep nanofinderparser`
2. Check that your Python scripts directory is in PATH
3. Try running with: `python -m nanofinderparser.cli`

## Development Issues

### Tests failing

If tests are failing in your development environment:

1. Install development dependencies: `uv sync`
2. Check Python version compatibility
3. Run tests individually to isolate issues:

   ```bash
   uv run pytest tests/test_specific.py -v
   ```

### Code style issues

If pre-commit hooks or CI checks are failing:

```bash
# Fix formatting
uv run ruff format .

# Fix linting issues
uv run ruff check --fix .

# Check types
uv run mypy src/nanofinderparser
```

### Environment issues

If you're having environment-related problems:

1. Use a clean virtual environment:
   ```bash
   uv venv --python 3.12
   uv sync
   ```

2. Clear any cached files:
   ```bash
   uv clean
   ```

## Performance Issues

### Slow imports

If imports are slow, check for:

1. Large dependencies being imported unnecessarily
2. Network calls during import
3. Heavy computation in module-level code

## Getting Help

If none of these solutions work:

1. Check existing [GitHub Issues](https://github.com/psolsfer/nanofinderparser/issues)
2. Create a new issue with:
   - Your operating system and Python version
   - Complete error messages
   - Steps to reproduce the problem
   - What you expected to happen
