# Codex CUDA Transformer

This repository provides paired Python (PyTorch) and C++ (LibTorch/CUDA) reference implementations for a family of small decoder-only Transformer language models that are convenient for experimentation and education.

## Model presets

Three presets expose progressively larger configurations while keeping the architecture fixed:

| name | parameters (approx.) | layers | model dim | heads | ffn dim | context |
|------|----------------------|--------|-----------|-------|---------|---------|
| `pico` | ~5M | 6 | 256 | 8 | 1024 | 256 |
| `nano` | ~50M | 12 | 512 | 8 | 2048 | 512 |
| `micro` | ~500M | 24 | 1024 | 16 | 8192 | 1024 |

Each preset can adjust the vocabulary size automatically to match the dataset tokenization.

## Python training

The Python implementation lives under [`python/codex_lm`](python/codex_lm). Install dependencies and launch training:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r python/requirements.txt
python -m codex_lm pico tinyshakespeare --steps 2000 --wandb
```

By default the script downloads the Tiny Shakespeare dataset and uses a character-level tokenizer. Use `--data custom --data-path /path/to/text.txt` to point at another corpus. Mixed precision, gradient accumulation, cosine scheduling, and optional `torch.compile` support are built in. Enable Weights & Biases logging with `--wandb` and optionally customize the project/run name via `--wandb-project` and `--wandb-run`.

## C++/CUDA training

The C++ training binary uses LibTorch with CUDA support. Configure CMake with an environment where `Torch_DIR` points to your LibTorch installation:

```bash
mkdir -p cpp/build
cd cpp/build
cmake -DCMAKE_PREFIX_PATH="${LIBTORCH_PATH}" ..
cmake --build .
./transformer_train --data ../../data/tinyshakespeare.txt --model pico --steps 2000 --wandb
```

The binary shares the same preset definitions and training tricks as the Python version. When `--wandb` is provided it streams metrics through the helper `codex_lm.wandb_stream` module, so ensure `python` dependencies are installed and available via `PYTHONPATH=../../python` when running the executable.

## Datasets

Text files are stored under the `data/` directory. The Python utilities can download Tiny Shakespeare automatically, while the C++ runner expects the file to exist locally. For larger experiments consider datasets such as Simple English Wikipedia once tokenization is adapted.

## Checkpoints

Training checkpoints are emitted to `checkpoints/` by the Python trainer and to `cpp_last_model.pt` by the C++ binary. Both contain standard PyTorch state dictionaries that can be loaded for evaluation or fine-tuning.

