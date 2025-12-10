# Contributing to Music Organizer

Thank you for your interest in contributing to Music Organizer! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:

- **Clear title**: Brief description of the bug
- **Description**: Detailed explanation of what happened
- **Steps to reproduce**: List of steps to trigger the bug
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Environment**: OS, Python version, tool version
- **Additional context**: Any relevant logs, screenshots, or files

### Suggesting Features

We welcome feature suggestions! Please:

1. Check existing issues to avoid duplicates
2. Create a new issue with the "enhancement" label
3. Include:
   - Clear description of the feature
   - Use case and motivation
   - Any implementation ideas
   - Possible UI/UX considerations

### Contributing Code

#### Prerequisites

- Python 3.11+
- astral uv package manager
- Familiarity with Git and GitHub

#### Setup

1. Fork the repository
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/music-organizer.git
   cd music-organizer
   ```

3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/yourusername/music-organizer.git
   ```

4. Install dependencies:
   ```bash
   uv install
   uv pip install -e .
   ```

#### Development Workflow

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes:
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. Run the test suite:
   ```bash
   uv run pytest
   ```

4. Check code quality:
   ```bash
   uv run black src/ tests/
   uv run ruff check src/ tests/
   uv run mypy src/
   ```

5. Commit your changes:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

6. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

7. Create a pull request

#### Code Style

- Use **Black** for formatting (automatically applied)
- Follow **PEP 8** guidelines
- Use **type hints** for all functions
- Write **descriptive** commit messages using conventional commits:
  - `feat:` for new features
  - `fix:` for bug fixes
  - `docs:` for documentation changes
  - `refactor:` for code refactoring
  - `test:` for test changes

#### Testing

- Write tests for all new functionality
- Ensure tests cover edge cases
- Use descriptive test names
- Add tests for any bugs you fix

#### Documentation

- Update README.md for user-facing changes
- Add docstrings to all public functions and classes
- Include examples in docstrings where helpful
- Update CHANGELOG.md for significant changes

## 🏗️ Architecture

The project is organized into several key modules:

### Core Components

- **`metadata.py`**: Handles audio metadata extraction using mutagen
- **`classifier.py`**: Determines content type (live, collaboration, etc.)
- **`organizer.py`**: Main orchestration logic
- **`mover.py`**: File operations with backup and safety features

### Models

- **`audio_file.py`**: Data model for audio files
- **`config.py`**: Configuration model with Pydantic

### CLI

- **`cli.py`**: Command-line interface using Click

## 📝 Development Guidelines

### Design Principles

1. **Safety First**: Never lose or corrupt user data
2. **Clear Errors**: Provide helpful error messages
3. **Performance**: Handle large libraries efficiently
4. **Flexibility**: Support various music organization preferences

### Adding New Features

When adding features:

1. Consider the impact on existing users
2. Ensure backward compatibility
3. Add appropriate tests
4. Update documentation

### File Organization

- Keep modules focused and single-purpose
- Use clear, descriptive names
- Group related functionality together

## 🧪 Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=music_organizer

# Run specific test file
uv run pytest tests/test_metadata.py

# Run tests matching pattern
uv run pytest -k "test_classify"
```

### Test Structure

- Unit tests in `tests/` directory
- One test file per module
- Test files named `test_*.py`
- Use descriptive test method names

### Fixtures

Test fixtures for audio files are in `tests/fixtures/`. Add new fixtures as needed for testing different file types and metadata scenarios.

## 📋 Pull Request Process

1. **Update Documentation**: Ensure your changes are documented
2. **Add Tests**: Include tests for new functionality
3. **Pass CI**: Ensure all checks pass
4. **One PR Per Feature**: Keep pull requests focused
5. **Clear Description**: Explain what you changed and why

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Added tests for new functionality
- [ ] All tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (if applicable)
```

## 🏷️ Labels

We use GitHub labels to categorize issues:

- `bug`: Bug reports
- `enhancement`: Feature requests
- `documentation`: Documentation issues
- `good first issue`: Good for newcomers
- `help wanted`: Need help
- `priority/high`: High priority issues

## 🚀 Release Process

Releases are handled by maintainers:

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create GitHub release with notes
4. Tag the release

## 💬 Getting Help

- **Discussions**: Use GitHub Discussions for questions
- **Issues**: Use GitHub Issues for bugs and features
- **Email**: Contact maintainers for sensitive issues

## 📄 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for everyone. Please:

- Be respectful and considerate
- Use inclusive language
- Focus on what is best for the community
- Show empathy toward other community members

### Unacceptable Behavior

- Harassment, trolling, or discrimination
- Publishing private information
- Spam or unnecessary promotion
- Disruptive behavior

### Enforcement

Project maintainers have the right and responsibility to remove, edit, or reject comments, commits, code, wiki edits, issues, and other contributions that are not aligned with this Code of Conduct.

## 🙏 Recognition

Contributors are recognized in:
- `README.md` contributors section
- Release notes
- GitHub contributors list

Thank you for contributing to Music Organizer! 🎵