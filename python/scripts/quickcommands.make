
train-pico:
	python -m codex_lm train \
		--tokenizer char \
		--data-frac 1.0 \
		--batch-size 16 --micro-batch-size 16 \
		--steps 5000 \
		--eval-interval 1 --eval-iters 1 \
		--device cuda \
		--wandb --wandb-project codex-transformer --wandb-run make-train-pico \
		--dtype float16 \
		pico tinyshakespeare

test-pico:
	python -m codex_lm generate \
		--prompt "HORATIO:" \
		--device cuda \
		./checkpoints/last.pt tinyshakespeare
