# SOO Fine-Tuning — Reproduction & Construct-Validity Testbed

A faithful, minimal reproduction of the LLM-track **Self-Other Overlap (SOO)
fine-tuning** experiment from Carauleanu et al. (2024), *["Towards Safe and
Honest AI Agents with Neural Self-Other Overlap"](https://arxiv.org/abs/2412.16325)*
(AE Studio; NeurIPS 2024 Safe Generative AI Workshop) — plus a testbed that
asks a question the original benchmark cannot answer: **does SOO induce
empathy, or does it install a flat no-lying rule?**

## Why this repo exists

SOO fine-tuning reduces deceptive responses in LLMs, with the drop attributed
to an empathy-like alignment of self- and other-representations. But the
paper's primary benchmark — "Burglar Bob" — has a single fixed property:
honesty hurts the model's self-interest and helps the recipient. On that
scenario, three very different mechanisms make *identical* predictions:

- **H1 — Flat no-lying rule.** A behavioral policy; no perspective-taking.
- **H2 — Recipient-indexed transparency.** Overlap with the interlocutor.
- **H3 — Genuine welfare overlap.** The paper's empathy claim.

The reduction in deception is real; the *explanation* is under-determined by
the evidence. This repo reproduces the effect and then supplies the scenarios
where H1/H2/H3 diverge, so the empathy claim becomes falsifiable.

## Contents

| File | What it does |
|------|--------------|
| `soo_finetune.py` | Reproduces SOO LoRA fine-tuning on Mistral-7B-Instruct-v0.2. Captures activations at `self_attn.o_proj` (layer 19), applies the MSE SOO loss over self/other prompt pairs, and reports Burglar Bob deception rate before/after. |
| `beyond_binary_bob.py` | The **"Beyond Binary Bob"** battery. Five cells that vary the *welfare valence of honesty* to separate H1/H2/H3, run against baseline and the SOO checkpoint, with a populated prediction matrix as output. |

### The battery

| Cell | Setup | Discriminates |
|------|-------|---------------|
| 1. Burglar Bob | Baseline; honesty helps recipient | anchor (all agree) |
| 2. Fire | Honesty sends Bob into a fire | H3 vs {H1,H2} |
| 3. Murderer Carl | The interlocutor is the threat | H2 vs H3 |
| 4. Portrait | Emotional white lie (gradient 0–5) | H3 vs {H1,H2} |
| 5. Delivery | Truth held constant; delivery only | H3 vs {H1,H2} |

Cells 4 and 5 are an **affective-empathy** probe. The paper's only
distinction-preserved check (the "Perspectives" scenario) is a Theory-of-Mind /
cognitive-empathy test — the faculty psychopaths retain — so it cannot speak to
the affective-empathy deficit the paper's own psychopathy motivation invokes.

## Quickstart

Requires a GPU (~10–14 GB VRAM for Mistral-7B in 4-bit + LoRA).

```bash
pip install "torch>=2.1" transformers peft bitsandbytes accelerate datasets anthropic
huggingface-cli login                     # Mistral-7B-Instruct-v0.2 is gated

# 1. Reproduce SOO fine-tuning
python soo_finetune.py --reduction mean

# 2. Run the Line 3 battery, baseline vs SOO-FT, side by side
export ANTHROPIC_API_KEY=...              # for gradient-cell judging (cells 4,5)
python beyond_binary_bob.py --adapter ./soo_mistral_lora --compare
```

## Reading the results

The decisive quantity is the **delta**, baseline → SOO-FT:

- Cells 2/3 move toward `welfare` → evidence for **H3** (empathy).
- Cells 2/3 stay `honest`-even-when-harmful → evidence for **H1** (no-lying rule).
- Cell 3 `welfare` while an ally-interlocutor case stays `honest` → **H2**.
- Cells 4/5 mean rises vs baseline → an affective-empathy signal.

## Known simplifications

This is a mechanism-faithful reference implementation, not a full replication.
For publishable numbers, add: 5-seed averaging, the full 7-scenario
generalization battery, MT-Bench for capability preservation, an LLM-as-judge
deception scorer with anchored exemplars, and the latent-SOO cross-layer MSE
metric. The `--reduction {mean,last}` flag exposes the sequence-pooling choice
left unspecified in the paper — worth running both, since a signal that
survives only under one reduction is riding on an anchor artifact.

## Citation

```bibtex
@inproceedings{carauleanu2024soo,
  title  = {Towards Safe and Honest AI Agents with Neural Self-Other Overlap},
  author = {Carauleanu, Marc and Vaiana, Michael and Rosenblatt, Judd and
            Berg, Cameron and Schwerz de Lucena, Diogo},
  year   = {2024},
  note   = {NeurIPS 2024 Safe Generative AI Workshop. arXiv:2412.16325}
}
```

## Disclaimer

Independent reproduction and critique for research purposes. Not affiliated with
or endorsed by the original authors. The goal is to strengthen a future
iteration of the method, not to discredit it.