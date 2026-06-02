# Contributing

Thank you for contributing to `model-quantization-recipes`. This repository is intended to provide reproducible, well-documented quantization workflows that can be reviewed and reused by others.

## Contribution scope

Contributions may include:

- New quantization recipes
- Improvements to existing scripts or documentation
- Reproducibility fixes
- Validation updates for supported environments

## Recipe requirements

New recipes should:

1. Be placed in a dedicated folder under `recipes/<recipe-name>/`.
2. Use `recipes/_template/README.md` as the documentation baseline.
3. Add a matching optional dependency group in the root `pyproject.toml` so the recipe installs with `uv sync --extra <recipe-name>`.
4. Document the model, quantization method, runtime target, hardware tested, dependencies, required environment variables, execution steps, and expected outputs.
5. Keep runnable entrypoints local to the recipe whenever practical.
6. Include enough validation detail for another contributor to reproduce the workflow.

## Documentation standards

- Keep instructions precise, current, and reproducible.
- Prefer explicit commands over implied setup steps.
- Record tested GPU, CUDA, Python, and framework versions when relevant.
- Avoid including credentials, access tokens, private URLs, or other sensitive internal information.

## Review checklist

Before opening a change for review, confirm that:

1. The recipe documentation matches the actual files and commands.
2. Required dependencies and environment variables are documented.
3. Example commands are complete and executable in the stated environment.
4. Output artifacts or validation results are described clearly.
5. Unrelated files are not modified.

## Pull request guidance

Keep pull requests focused on a single recipe or a closely related documentation change. Summaries should describe the purpose of the change, the validation performed, and any known limitations or follow-up work.
