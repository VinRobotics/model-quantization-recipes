# <recipe-name>

Short description of the model, quantization method, and target runtime.

## Status

- Owner:
- Source branch or repo:
- Migration state: template

## Scope

- Model:
- Quantization method:
- Runtime or deployment target:
- Hardware tested:

## Files expected in this folder

- `README.md`
- `pyproject.toml` optional dependency group for the recipe
- `scripts/` or runnable `.sh` wrappers
- Python entrypoints for quantization, export, benchmark, or inference

## Environment

Document all required environment variables here.

Example:

```bash
export MODEL_PATH=/path/to/model
export OUTPUT_PATH=/path/to/output
```

## Quick start

```bash
uv sync --extra <recipe-name>
python <entrypoint>.py --help
```

## Validation

Define the minimum validation required after migration:

- script starts
- dependencies install cleanly
- output directory is created
- artifacts are generated as expected

## Notes for migration

- Preserve original filenames first.
- Record upstream assumptions that are currently hardcoded.
- Add sample commands that work on a fresh machine.
