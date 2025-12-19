# HF Example Repo

This is a simple example project using Hugging Face transformers and nnsight to extract layers.

## Setup

This project uses `uv` for Python dependency management and virtual environments.

### Prerequisites

- uv (https://github.com/indygreg/uv)

### How to run

To install dependencies:

```bash
uv sync
```

To install development dependencies (for testing and formatting):

```bash
uv sync --extra dev --extra test
```

To install all dependencies (project, development, and testing):

```bash
uv sync --all-extras
```

To run the main project:

```bash
uv run hf_example_repo
```

To run the activation grabber example:

```bash
uv run hf_activation_grabber_example
```

Other available scripts:

- `uv run hf_examples`
- `uv run hf_gpt2_example`
- `uv run hf_working_gpt2_example`
- `uv run hf_minimal_gpt2_example`
- `uv run hf_debug_gpt2_config`

### Testing

To run all tests:

```bash
uv run pytest tests/
```

To run tests with coverage:

```bash
uv run pytest tests/ --cov=hf_example_repo --cov-report=html
```

### Code Formatting

To format the code using ruff:

```bash
uv run ruff format .
```

To check for linting issues:

```bash
uv run ruff check .
```

## Hallucination Probe

The probe pipeline detects hallucinations by analyzing the model's internal state at the moment it begins generating a response. The key insight is that the model's hidden state at the **last token of the prompt** (right before generation starts) contains information about whether the upcoming response will be correct or hallucinated.

The pipeline:
1. Takes questions with known correct answers
2. Generates model responses and labels them as correct/hallucinated
3. Extracts activations at the **BOS of response** position (last token of formatted prompt)
4. Trains a logistic regression classifier on these activations

Once trained, the probe can predict hallucination likelihood for new queries by extracting activations at the same position.

### Quick Start

```bash
uv run hf_example_repo
```

This will:
1. Label raw NQ dataset (if not already labeled)
2. Train a probe on the labeled data (if not already trained)
3. Run the probe on a sample query

### Accessing Activations

Example use case on how to extract activations.

```python
from hf_example_repo.activiation_grabber import ActivationGrabber, ActivationConfig
from hf_example_repo.subject import Subject, get_subject_config

subject = Subject(config=get_subject_config("meta-llama/Llama-2-7b-chat-hf"), ...)
activation_grabber = ActivationGrabber(subject)

# Get activations for a text
activation_data = activation_grabber.get_activations("Your text here")

# Shape: (batch, layers, tokens, hidden_dim)
activations = activation_data.activations

# Access specific layer/token
layer_8_last_token = activations[0, 8, -1, :]  # shape: (hidden_dim,)
```

### Provided Dataset

The raw Natural Questions (NQ) dataset is provided at `src/probe/data/nq_dataset.json` with question-answer pairs.

On first run, the pipeline automatically:
1. **Labels the dataset** → generates model responses, compares to ground truth, saves to `artifacts/labeled_nq.json`
2. **Trains the probe** → extracts activations, trains classifier, saves to `artifacts/probes/`

Subsequent runs reuse the existing labeled data and probe unless `force_relabel=True` is set.

### hal_utils Module

`src/probe/hal_utils.py` provides the core probe functionality:

- **Dataset loading**: `load_nq_dataset()` loads raw NQ data; `load_labeled_json()` loads labeled data
- **Labeling**: `evaluate_model_on_dataset()` generates responses and labels them using `symmetric_verify_answer()` (word overlap comparison)
- **Feature extraction**: `extract_features()` uses `ActivationGrabber` to get hidden states at a specified layer/position
- **Training**: `train_probe()` fits a StandardScaler + LogisticRegression on activations, saves artifacts
- **Inference**: `predict_with_probe()` loads a trained probe and returns predictions/probabilities
- **Orchestration**: `ensure_probe()` and `run_probe_on_queries()` handle the full pipeline (label if needed → train if needed → predict)

### Labeled Dataset Format

The probe expects a JSON file with `data` array containing labeled examples:

```json
{
  "metadata": { ... },
  "data": [
    {
      "question": "who is the coach of ohio state football?",
      "ground_truth": "Urban Meyer",
      "model_response": "Ryan Day",
      "is_correct": 0,
      "overlap_fraction": 0.0
    },
    {
      "question": "where are presidential powers listed in the constitution?",
      "ground_truth": "Article II",
      "model_response": "Article II, Section 1 of the US",
      "is_correct": 1,
      "overlap_fraction": 1.0
    }
  ]
}
```

**Required fields for probe training:**
- `question`: The input text fed to the model
- `is_correct`: Label - `1` = model answered correctly, `0` = hallucination

**Labeling logic:** The pipeline generates model responses to questions and compares them to ground truth using word overlap. If ≥50% of ground truth words appear in the response, it's labeled correct (`1`); otherwise hallucination (`0`). Ambiguous cases (0-50% overlap) are discarded.

### Future Improvements

- **Model loading**: Currently, the labeling step loads the model via standard HuggingFace `AutoModelForCausalLM`. In the future, this could be unified to use `Subject` for both labeling and activation extraction.

