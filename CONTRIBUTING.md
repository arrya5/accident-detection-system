# Contributing to the Accident Detection System

Thanks for your interest in contributing! 🎉 This project is an open-source, AI-powered road accident detection and alert system, and we welcome contributions of all kinds — bug fixes, model improvements, documentation, new alert channels, and more.

## Ways to contribute

- 🐛 **Report a bug** — open an issue with steps to reproduce and your environment details.
- 💡 **Suggest a feature** — e.g. a new notification channel (SMS, webhook), or a different model backbone.
- 📚 **Improve docs** — clarify setup steps, add troubleshooting notes, or improve the README.
- 🧠 **Improve the model** — better augmentation, alternative architectures, or reduced false positives.
- 🛠️ **Submit code** — pick an open issue (ideally labeled `good first issue`) and send a pull request.

## Development setup

See the [README](README.md) for full details. In short:

```bash
# Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run detection on a video source
python src/detect_pytorch.py --source video.mp4 --email
```

> A CUDA-capable NVIDIA GPU is recommended for real-time (25+ FPS) performance, but the system also runs on CPU at lower frame rates.

## Pull request process

1. **Fork** the repository and create a branch from `main`:
   ```bash
   git checkout -b fix/short-description
   ```
2. **Make your change.** Keep each PR focused on one logical change.
3. **Don't commit large files.** Do not push model weights, datasets, or video files — keep them out via `.gitignore`.
4. **Test locally** on a sample video or image to confirm detection still works.
5. **Commit** with a clear message:
   ```
   feat: add SMS alert channel via Twilio
   ```
6. **Push** and open a Pull Request against `main`, referencing any related issue (e.g. `Fixes #7`).
7. Respond to review feedback — a maintainer will merge once it's ready.

## Code style

- Follow **PEP 8** for Python.
- Keep inference, preprocessing, and alerting logic in separate, testable functions.
- Use descriptive variable names; reserve comments for non-obvious logic.

## Reporting an issue

When opening an issue, please include:
- What you expected vs. what happened
- Steps to reproduce
- Your OS, Python version, and whether you're running on GPU or CPU
- Relevant logs, screenshots, or sample frames

## Code of conduct

Be respectful and constructive. This project aims to be a welcoming place for first-time open-source contributors.

---

Questions? Open an issue or reach out to [@arrya5](https://github.com/arrya5).
