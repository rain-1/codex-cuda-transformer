# Codex CUDA Transformer

This repository provides paired Python (PyTorch) and C++ (LibTorch/CUDA) reference implementations for a family of small decoder-only Transformer language models that are convenient for experimentation and education.

## Model presets

Three presets expose progressively larger configurations while keeping the architecture fixed:

| name | parameters (approx.) | layers | model dim | heads | ffn dim | context |
|------|----------------------|--------|-----------|-------|---------|---------|
| `pico` | ~5M | 6 | 256 | 8 | 1024 | 256 |
| `nano` | ~50M | 12 | 512 | 8 | 2048 | 512 |
| `micro` | ~200M | 16 | 640 | 16 | 3328 | 1024 |

Each preset can adjust the vocabulary size automatically to match the dataset tokenization.

## Python training

The Python implementation lives under [`python/codex_lm`](python/codex_lm). Install dependencies and launch training:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r python/requirements.txt
python -m codex_lm train pico tinyshakespeare --steps 2000 --wandb
```

To train on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) instead, swap the dataset argument:

```bash
python -m codex_lm train pico tinystories --steps 2000 --wandb
```

Switch to a word-and-punctuation tokenizer by adding `--tokenizer word`:

```bash
python -m codex_lm train pico tinystories --tokenizer word --steps 2000 --wandb
```

See preset details (layers, dimensions, parameter counts) without launching training:

```bash
python -m codex_lm info nano
```

If you only need a smaller subset of a large corpus (to reduce RAM or download time), limit the amount of text consumed via `--data-frac`, e.g.:

```bash
python -m codex_lm train pico tinystories --data-frac 0.1 --steps 2000 --wandb
```

By default the script downloads the Tiny Shakespeare dataset and uses a character-level tokenizer. Use `--data custom --data-path /path/to/text.txt` to point at another corpus. Mixed precision, configurable gradient accumulation (`--gradient-accumulation-steps`), gradient checkpointing (`--gradient-checkpointing`), cosine scheduling, and optional `torch.compile` support are built in. Enable Weights & Biases logging with `--wandb` and optionally customize the project/run name via `--wandb-project` and `--wandb-run`.

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

Text files are stored under the `data/` directory. The Python utilities can download Tiny Shakespeare automatically, while the C++ runner expects the file to exist locally. TinyStories downloads are pulled directly from Hugging Face (training and validation splits) and merged into a single `tinystories.txt` file with `<|endoftext|>`/`<|end_of_sequence|>` markers replaced by blank lines. When a dataset exceeds ~256 MiB it is automatically converted into a disk-backed token cache (`*.tokens.npy` + metadata, keyed by tokenizer choice) so that training can stream batches without loading every token into RAM; the first run will create this cache and later runs reuse it. Use `--data-frac <fraction>` (e.g., `0.1` for 10 %) to keep only the leading portion of massive corpora, and `--tokenizer word` to tokenize into words/punctuation instead of raw characters. For larger experiments consider datasets such as Simple English Wikipedia once tokenization is adapted.

## Checkpoints

Training checkpoints are emitted to `checkpoints/` by the Python trainer and to `cpp_last_model.pt` by the C++ binary. Both contain standard PyTorch state dictionaries that can be loaded for evaluation or fine-tuning.

