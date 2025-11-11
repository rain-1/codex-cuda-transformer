
train-pico:
	python -m codex_lm train \
		--tokenizer char \
		--data-frac 1.0 \
		--batch-size 16 --micro-batch-size 16 \
		--steps 100 \
		--eval-interval 10 --eval-iters 1 \
		--device cuda \
		--wandb --wandb-project codex-transformer --wandb-run make-train-pico \
		--dtype float16 \
		pico tinyshakespeare

test-pico:
	python -m codex_lm generate \
		--prompt "HORATIO:" \
		--device cuda \
		./checkpoints/last.pt tinyshakespeare


train-nano:
	python -m codex_lm train \
		--tokenizer word \
		--data-frac 0.2 \
		--batch-size 8 --micro-batch-size 8 \
		--steps 100 \
		--eval-interval 10 --eval-iters 1 \
		--device cuda \
		--wandb --wandb-project codex-transformer --wandb-run make-train-nano \
		--dtype float16 \
		nano tinystories

train-nano-gradcheck:
	python -m codex_lm train \
		--tokenizer word \
		--data-frac 0.2 \
		--batch-size 8 --micro-batch-size 8 \
		--steps 100 \
		--eval-interval 10 --eval-iters 1 \
		--device cuda \
		--wandb --wandb-project codex-transformer --wandb-run make-train-nano-gradcheck \
		--dtype float16 \
		--gradient-checkpointing \
		nano tinystories

train-nano-gradopts:
	python -m codex_lm train \
		--tokenizer word \
		--data-frac 0.2 \
		--batch-size 8 --micro-batch-size 8 \
		--steps 100 \
		--eval-interval 10 --eval-iters 1 \
		--device cuda \
		--wandb --wandb-project codex-transformer --wandb-run make-train-nano-gradopt \
		--dtype float16 \
		--gradient-checkpointing --gradient-accumulation-steps 8 \
		nano tinystories
