# AG News: in-context learning and LoRA

CMU 11-667 mini-project comparing Qwen2.5-0.5B-Instruct and SmolLM2-360M-Instruct on four-class news classification. Each model is evaluated with validation-selected prompts and LoRA adapters. A matched four-shot experiment tests whether TF-IDF retrieval improves example selection.

Prompt development compares demonstration count, category definitions, an explicit final output instruction and alternating chat-turn demonstrations. The latter formats were added after validation exposed failures in the original completion-style prompt.

## Run

Python 3.12 was used. Install a CUDA-enabled PyTorch build suitable for your machine, then the remaining dependencies:

```bash
python -m pip install -r requirements.txt
python experiment.py prepare
python experiment.py develop
python experiment.py test
python experiment.py analyze
python make_report.py
```

To rebuild the PDF, install `requirements-report.txt` and run `python render_report.py`. The renderer uses Arial on Windows or DejaVu Sans on Linux.

The full run uses 2,000 training, 400 validation and 400 fresh test examples. `PROTOCOL.json` records the search space and split rules. Base-model revisions are pinned in `experiment.py`; models and AG News are downloaded on first use. Set `CMU_MODEL_CACHE` to reuse model weights, or `HF_HOME` to change the dataset cache. Use `--offline` once all files are cached. Training is intended for a CUDA GPU, with BF16 on supported hardware.

Run development before test evaluation. The selected configurations are hashed before test prediction; subsequent development is blocked. Cached evaluations check configuration and sample IDs. Code changes require a new results directory because cache keys do not include a source hash.

## Contents

- `experiment.py`: prompt search, LoRA training, final evaluation and analysis.
- `prepare_data.py`: deduplication and reproducible stratified splits.
- `make_report.py`: report built from saved predictions and training logs.
- `results/`: final predictions, validation trials, figures and configuration records.
- `report.md`, `report.pdf`: methods, results, error analysis and deployment discussion.
- `presentation.pptx`: short presentation of the experiment and findings.

Adapters and base weights are not stored in Git. To retrain without changing the published results, run `python experiment.py all --results-dir reproduction`. Training logs are retained separately from the weights. The small pilot from the earlier version is separate from this experiment. Its test sample is excluded from the new test set.

## Evaluation

Models generate one of `World`, `Sports`, `Business`, `Sci/Tech`. After trimming whitespace and ignoring case, the entire generated string must match a label. Extra explanations and unknown labels count as errors. Accuracy, macro-F1, per-class scores, invalid output rate, confusion matrices and paired comparisons are reported. Exact label matching deliberately exposes failures to follow the requested output format.

```bash
python -m unittest test_evaluation -v
python test_training_loss.py
python verify_results.py
```

The first command checks scoring and split isolation. The second compares the memory-saving answer-only loss and LoRA gradients against standard causal-LM loss on small Qwen2 and Llama models. The third checks final predictions against their raw outputs, recalculates metrics and verifies the frozen configurations.

## Sources

- [AG News dataset](https://huggingface.co/datasets/fancyzhx/ag_news)
- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)
- [Hu et al., LoRA](https://arxiv.org/abs/2106.09685)

AG News contains older English-language news and does not represent current multilingual news traffic. Exact deduplication does not remove semantic near-duplicates, and pretraining overlap has not been audited. Model-family comparisons also differ in parameter count and training history. This is a single-seed study, not a claim of production performance.

Other projects: [Environmental Language Model](https://github.com/Yulin2628/environmental-language-model) and [AI Buyer Assistant](https://github.com/Yulin2628/ai-buyer-assistant).
