"""
Single-file HAL probe utilities: dataset loading, feature extraction,
artifact IO, training, and inference.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

from hf_example_repo.activiation_grabber import ActivationConfig, ActivationGrabber
from hf_example_repo.subject import Subject

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _normalize_records(raw: Any) -> List[dict]:
    if isinstance(raw, dict) and "data" in raw:
        raw = raw["data"]
    if not isinstance(raw, list):
        raw = list(raw)
    return raw


def load_labeled_json(
    file_path: str | Path,
    text_key: str = "question",
    label_key: str = "label",
    max_samples: int | None = None,
) -> Tuple[List[str], List[int]]:
    """
    Load a JSON dataset with text/label keys. Supports top-level list or {"data": [...]}.
    """
    path = Path(file_path)
    with path.open("r") as f:
        raw = json.load(f)

    records = _normalize_records(raw)
    if max_samples is not None:
        records = records[:max_samples]

    texts: List[str] = []
    labels: List[int] = []
    for rec in records:
        if text_key not in rec or label_key not in rec:
            raise KeyError(f"Record missing required keys: {text_key}, {label_key}")
        texts.append(str(rec[text_key]))
        labels.append(int(rec[label_key]))

    return texts, labels


# Natural Questions loader (raw triples -> dicts)
def load_nq_dataset(file_path: str | Path, max_samples: int | None = None) -> List[dict]:
    """
    Load Natural Questions-style dataset: each entry is [question, answer, token_list].
    """
    path = Path(file_path)
    with path.open("r") as f:
        data = json.load(f)

    if max_samples:
        data = data[:max_samples]

    nq_data: List[dict] = []
    for item in data:
        nq_data.append(
            {
                "question": item[0].replace("question: ", "").replace("\nanswer:", ""),
                "answer": item[1],
                "tokens": item[2],
            }
        )
    return nq_data


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _select_token_position(attn_mask_row, pos_mode: str, pos_idx: Optional[int]) -> int:
    seq_len = int(sum(attn_mask_row))
    if seq_len <= 0:
        return 0
    if pos_mode in {"last", "eos"}:
        pos = seq_len - 1
    elif pos_mode == "idx":
        if pos_idx is None:
            raise ValueError("pos_idx must be provided when pos_mode='idx'")
        pos = pos_idx
    else:
        raise ValueError(f"Unsupported pos_mode={pos_mode}")
    return max(0, min(pos, seq_len - 1))


def _pool_tokens(hs_LTI: np.ndarray, pool: str, attn_mask_row) -> np.ndarray:
    if pool == "none":
        return hs_LTI
    if pool == "mean":
        seq_len = int(sum(attn_mask_row))
        seq_len = max(seq_len, 1)
        return hs_LTI[:, :seq_len, :].mean(axis=1)
    raise ValueError(f"Unsupported pool={pool}")


def extract_features(
    subject: Subject,
    activation_grabber: ActivationGrabber,
    texts: List[str],
    layer_idx: int,
    pos_mode: str = "last",
    pos_idx: Optional[int] = None,
    pool: str = "none",
) -> np.ndarray:
    """
    Extract hidden-state features from a given layer for a list of texts.
    """
    if subject.tokenizer.pad_token is None:
        subject.tokenizer.pad_token = subject.tokenizer.eos_token

    config = ActivationConfig(layers=[layer_idx], return_numpy=True)
    acts = activation_grabber.get_activations(texts, config=config)

    hs_BLTI = acts.activations
    attn = acts.attention_mask

    feats = []
    for b in range(len(texts)):
        pos = _select_token_position(attn[b], pos_mode, pos_idx)
        token_vec = hs_BLTI[b, 0:1, pos : pos + 1, :]
        pooled = _pool_tokens(token_vec, pool, attn[b])
        feats.append(pooled.reshape(-1))

    return np.stack(feats, axis=0)


# ---------------------------------------------------------------------------
# Probe artifact IO
# ---------------------------------------------------------------------------


DEFAULT_PROBE_DIR = Path("artifacts/probes")


@dataclass
class ProbeArtifacts:
    base_path: Path
    scaler_path: Path
    clf_path: Path
    metrics_path: Path
    meta_path: Path
    metrics: Dict[str, Any] | None = None
    meta: Dict[str, Any] | None = None


def build_probe_base_name(model_name: str, layer_idx: int, pos_mode: str, tag: str) -> str:
    model = model_name.lower().replace("-", "_")
    return f"probe_{model}_L{layer_idx}_{pos_mode}_{tag}"


def save_probe_artifacts(
    out_dir: Path,
    base_name: str,
    scaler: Any,
    clf: Any,
    metrics: Dict[str, Any],
    meta: Dict[str, Any],
) -> ProbeArtifacts:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_path = out_dir / base_name
    scaler_path = out_dir / f"{base_name}_scaler.pkl"
    clf_path = out_dir / f"{base_name}_clf.pkl"
    metrics_path = out_dir / f"{base_name}_metrics.json"
    meta_path = out_dir / f"{base_name}_meta.json"

    with scaler_path.open("wb") as f:
        pickle.dump(scaler, f)
    with clf_path.open("wb") as f:
        pickle.dump(clf, f)
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    return ProbeArtifacts(base_path, scaler_path, clf_path, metrics_path, meta_path, metrics, meta)


def load_probe_artifacts(base_path: str | Path) -> Tuple[Any, Any, Dict[str, Any], Dict[str, Any]]:
    base = Path(base_path)
    scaler_path = base.with_name(base.name + "_scaler.pkl")
    clf_path = base.with_name(base.name + "_clf.pkl")
    metrics_path = base.with_name(base.name + "_metrics.json")
    meta_path = base.with_name(base.name + "_meta.json")

    with scaler_path.open("rb") as f:
        scaler = pickle.load(f)
    with clf_path.open("rb") as f:
        clf = pickle.load(f)
    with metrics_path.open("r") as f:
        metrics = json.load(f)
    with meta_path.open("r") as f:
        meta = json.load(f)

    return scaler, clf, metrics, meta


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class TrainProbeResult:
    artifacts: ProbeArtifacts
    metrics: Dict[str, Dict[str, float]]
    meta: Dict[str, Any]


def _split_data(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_train = int(len(X) * train_ratio)
    n_val = int(len(X) * val_ratio)
    idx = np.arange(len(X))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    tr_idx = idx[:n_train]
    va_idx = idx[n_train : n_train + n_val]
    te_idx = idx[n_train + n_val :]
    return X[tr_idx], X[va_idx], X[te_idx], y[tr_idx], y[va_idx], y[te_idx]


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "count": int(len(y_true)),
    }


def train_probe(
    texts: List[str],
    labels: List[int],
    subject: Subject,
    activation_grabber: ActivationGrabber,
    *,
    layer_idx: int,
    pos_mode: str = "last",
    pos_idx: Optional[int] = None,
    pool: str = "none",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    tag: str = "base",
    out_dir: Path = DEFAULT_PROBE_DIR,
) -> TrainProbeResult:
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")
    if len(texts) < 5:
        raise ValueError("Need at least 5 samples to train/validate/test a probe")

    X = extract_features(
        subject,
        activation_grabber,
        texts,
        layer_idx=layer_idx,
        pos_mode=pos_mode,
        pos_idx=pos_idx,
        pool=pool,
    )
    y = np.asarray(labels, dtype=np.int64)

    X_tr, X_va, X_te, y_tr, y_va, y_te = _split_data(X, y, train_ratio, val_ratio, seed)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)

    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    clf.fit(X_tr_s, y_tr)

    metrics = {
        "train": _compute_metrics(y_tr, clf.predict(X_tr_s)),
        "val": _compute_metrics(y_va, clf.predict(X_va_s)),
        "test": _compute_metrics(y_te, clf.predict(X_te_s)),
    }

    base_name = build_probe_base_name(subject.model_name, layer_idx, pos_mode, tag)
    meta = {
        "model_name": subject.model_name,
        "layer_idx": layer_idx,
        "pos_mode": pos_mode,
        "pos_idx": pos_idx,
        "pool": pool,
        "seed": seed,
        "tag": tag,
        "timestamp": int(time.time()),
        "counts": {
            "train": len(y_tr),
            "val": len(y_va),
            "test": len(y_te),
            "total": len(y),
        },
    }

    artifacts = save_probe_artifacts(out_dir, base_name, scaler, clf, metrics, meta)
    return TrainProbeResult(artifacts=artifacts, metrics=metrics, meta=meta)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict_with_probe(
    texts: List[str],
    subject: Subject,
    activation_grabber: ActivationGrabber,
    probe_base_path: str,
) -> Dict[str, Any]:
    scaler, clf, metrics, meta = load_probe_artifacts(probe_base_path)

    X = extract_features(
        subject,
        activation_grabber,
        texts,
        layer_idx=meta["layer_idx"],
        pos_mode=meta.get("pos_mode", "last"),
        pos_idx=meta.get("pos_idx"),
        pool=meta.get("pool", "none"),
    )
    X_norm = scaler.transform(X)
    preds = clf.predict(X_norm)
    probs = clf.predict_proba(X_norm)

    return {
        "preds": preds,
        "probs": probs,
        "meta": meta,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Labeling / hallucination detection pipeline
# ---------------------------------------------------------------------------

system_msg_qa = (
    "Always respond to the input question concisely with a short phrase or a single-word answer. "
    "Do not repeat the question or provide any explanation."
)


def format_prompt_for_probe(question: str, tokenizer, premise: str | None = None) -> str:
    """
    Format a question into the same prompt format used during labeling.
    This ensures probe training and inference use consistent input format.
    """
    if premise:
        question = f"{premise}\n{question}"
    messages = [{"role": "system", "content": system_msg_qa}, {"role": "user", "content": question}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system_msg_qa}\nQ: {question}\nA:"


def get_model_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    max_new_tokens: int = 10,
    premise: str | None = None,
) -> str:
    """
    Generate a short answer to a question using the given model/tokenizer.
    """
    if premise:
        question = f"{premise}\n{question}"

    messages = [{"role": "system", "content": system_msg_qa}, {"role": "user", "content": question}]
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"{system_msg_qa}\nQ: {question}\nA:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return response.strip()


def symmetric_verify_answer(model_answer: str, correct_answer: str, completeness_threshold: float = 0.5) -> tuple[bool, float, bool]:
    """
    Symmetric answer verification: returns (is_correct, overlap_fraction, should_delete).
    Strips punctuation from words before comparison to handle cases like "Superman." vs "Superman".
    """
    model_answer = model_answer.lower().strip()
    correct_answer = correct_answer.lower().strip()

    if len(correct_answer) == 0:
        completeness_score = 1.0 if len(model_answer) == 0 else 0.0
    else:
        # Strip punctuation from words before comparison
        def strip_punct(word: str) -> str:
            return re.sub(r'[^\w\s]', '', word)
        
        correct_words = {strip_punct(w) for w in correct_answer.split()}
        model_words = {strip_punct(w) for w in model_answer.split()}
        # Remove empty strings that might result from punctuation-only tokens
        correct_words = {w for w in correct_words if w}
        model_words = {w for w in model_words if w}
        
        completeness_score = len(correct_words.intersection(model_words)) / len(correct_words) if len(correct_words) else 0.0

    overlap_fraction = completeness_score
    is_correct = completeness_score >= completeness_threshold
    should_delete = 0 < completeness_score < 0.5
    return is_correct, overlap_fraction, should_delete


def evaluate_model_on_dataset(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: List[dict],
    model_name: str,
    completeness_threshold: float = 0.5,
    premise: str | None = None,
    max_samples: int | None = None,
) -> List[dict]:
    """
    Generate responses, verify with symmetric overlap, and produce labeled results.
    """
    results: List[dict] = []
    deleted_count = 0

    data_iter = dataset if max_samples is None else dataset[:max_samples]
    for i, item in enumerate(data_iter):
        if i % 100 == 0:
            print(f"Evaluating {i}/{len(data_iter)}...")

        question = item["question"]
        ground_truth = item["answer"]
        response = get_model_response(model, tokenizer, question, premise=premise)

        is_correct, overlap_fraction, should_delete = symmetric_verify_answer(response, ground_truth, completeness_threshold)
        if should_delete:
            deleted_count += 1
            continue

        results.append(
            {
                "question": question,
                "ground_truth": ground_truth,
                "ground_truth_tokens": item.get("tokens", []),
                "model_response": response,
                "is_correct": 1 if is_correct else 0,
                "overlap_fraction": overlap_fraction,
            }
        )

    print(f"Deleted {deleted_count} entries due to low completeness (< {completeness_threshold})")
    return results


def save_evaluation_results(results: List[dict], output_path: str | Path, metadata: dict | None = None) -> None:
    """
    Save labeled evaluation results to JSON in the format expected by probe training.
    """
    payload = {
        "metadata": metadata or {},
        "data": [
            {
                "index": i,
                "question": r["question"],
                "ground_truth": r["ground_truth"],
                "ground_truth_tokens": r.get("ground_truth_tokens", []),
                "model_response": r["model_response"],
                "is_correct": r["is_correct"],
                "overlap_fraction": r["overlap_fraction"],
            }
            for i, r in enumerate(results)
        ],
    }

    os.makedirs(Path(output_path).parent, exist_ok=True)
    with Path(output_path).open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved evaluation results to {output_path}")
    print(f"Total samples: {len(results)}")
    print(f"Correct: {sum(1 for r in results if r['is_correct'] == 1)}")
    print(f"Hallucinations: {sum(1 for r in results if r['is_correct'] == 0)}")


# ---------------------------------------------------------------------------
# Orchestration helpers (label -> train -> infer)
# ---------------------------------------------------------------------------


@dataclass
class ProbePipelineConfig:
    raw_dataset_path: Path
    labeled_path: Path
    probe_dir: Path
    label_model_id: str = "gpt2"
    label_premise: str | None = None
    label_max_samples: int | None = 200
    completeness: float = 0.5
    probe_layer: int = 8
    probe_pos_mode: str = "last"
    probe_pos_idx: int | None = None
    probe_pool: str = "none"
    probe_tag: str = "base"
    label_key: str = "is_correct"
    text_key: str = "question"
    force_relabel: bool = False  # If True, re-label even if labeled_path exists


def label_dataset_if_missing(cfg: ProbePipelineConfig) -> Path | None:
    """
    Label the raw dataset if the labeled file does not exist.
    Returns the path to the labeled dataset, or None on failure.
    """
    if cfg.labeled_path.exists() and not cfg.force_relabel:
        print(f"[Probe] Using existing labeled data: {cfg.labeled_path}")
        return cfg.labeled_path
    if cfg.force_relabel and cfg.labeled_path.exists():
        print(f"[Probe] force_relabel=True, re-labeling dataset (will overwrite {cfg.labeled_path})")
    if not cfg.raw_dataset_path.exists():
        print(f"[Probe] Raw dataset not found: {cfg.raw_dataset_path}")
        return None

    print(f"[Probe] Loading raw dataset from {cfg.raw_dataset_path}")
    raw = load_nq_dataset(cfg.raw_dataset_path, max_samples=cfg.label_max_samples)
    print(f"[Probe] Raw samples loaded: {len(raw)}")

    print(f"[Probe] Loading labeling model: {cfg.label_model_id}")
    tok = AutoTokenizer.from_pretrained(cfg.label_model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if torch.cuda.is_available() else None
    mdl = AutoModelForCausalLM.from_pretrained(cfg.label_model_id, torch_dtype=dtype, device_map="auto")

    print("[Probe] Evaluating and labeling...")
    results = evaluate_model_on_dataset(
        model=mdl,
        tokenizer=tok,
        dataset=raw,
        model_name=cfg.label_model_id,
        completeness_threshold=cfg.completeness,
        premise=cfg.label_premise,
        max_samples=cfg.label_max_samples,
    )
    metadata = {
        "model": cfg.label_model_id,
        "eval_samples": len(results),
        "completeness_threshold": cfg.completeness,
    }
    cfg.labeled_path.parent.mkdir(parents=True, exist_ok=True)
    save_evaluation_results(results, cfg.labeled_path, metadata=metadata)
    return cfg.labeled_path


def train_probe_if_missing(
    cfg: ProbePipelineConfig,
    subject: Subject,
    activation_grabber: ActivationGrabber,
) -> Path | None:
    """
    Train a probe if artifacts are missing or if labeled dataset is newer than probe artifacts.
    Returns base path or None.
    """
    cfg.probe_dir.mkdir(parents=True, exist_ok=True)
    base_name = build_probe_base_name(subject.model_name, cfg.probe_layer, cfg.probe_pos_mode, cfg.probe_tag)
    base_path = cfg.probe_dir / base_name
    clf_path = base_path.with_name(base_name + "_clf.pkl")
    
    # Check if probe exists and if labeled dataset is newer or force_relabel is set
    if clf_path.exists() and not cfg.force_relabel:
        if cfg.labeled_path.exists():
            labeled_mtime = cfg.labeled_path.stat().st_mtime
            probe_mtime = clf_path.stat().st_mtime
            if labeled_mtime > probe_mtime:
                print(f"[Probe] Labeled dataset is newer than probe artifacts, retraining probe...")
            else:
                print(f"[Probe] Using existing probe: {base_path}")
                return base_path
        else:
            print(f"[Probe] Using existing probe: {base_path}")
            return base_path
    elif clf_path.exists() and cfg.force_relabel:
        print(f"[Probe] force_relabel=True, retraining probe (will overwrite existing artifacts)...")

    if not cfg.labeled_path.exists():
        print(f"[Probe] Labeled dataset missing: {cfg.labeled_path}")
        return None

    texts, labels = load_labeled_json(
        cfg.labeled_path,
        text_key=cfg.text_key,
        label_key=cfg.label_key,
        max_samples=cfg.label_max_samples,
    )
    # Format questions with chat template to match labeling format
    # This ensures activations are extracted at the same position (BOS of response)
    formatted_texts = [format_prompt_for_probe(t, subject.tokenizer, cfg.label_premise) for t in texts]
    print(f"[Probe] Training probe on {len(formatted_texts)} samples (formatted with chat template)")
    result = train_probe(
        formatted_texts,
        labels,
        subject,
        activation_grabber,
        layer_idx=cfg.probe_layer,
        pos_mode=cfg.probe_pos_mode,
        pos_idx=cfg.probe_pos_idx,
        pool=cfg.probe_pool,
        tag=cfg.probe_tag,
        out_dir=cfg.probe_dir,
    )
    print("[Probe] Trained probe:")
    print(f"  Artifacts: {result.artifacts.base_path}")
    print(f"  Data split: train={result.meta['counts']['train']}, val={result.meta['counts']['val']}, test={result.meta['counts']['test']}")
    print("[Probe] Validation results:")
    for split_name in ["train", "val", "test"]:
        m = result.metrics[split_name]
        print(f"  {split_name:5s}: acc={m['acc']:.3f}, f1={m['f1']:.3f}, n={m['count']}")
    return result.artifacts.base_path


def ensure_probe(cfg: ProbePipelineConfig, subject: Subject, activation_grabber: ActivationGrabber) -> Path | None:
    """
    Ensure labeled data and probe exist; return probe base path or None.
    """
    labeled_path = label_dataset_if_missing(cfg)
    if labeled_path is None:
        return None
    return train_probe_if_missing(cfg, subject, activation_grabber)


def run_probe_on_queries(
    cfg: ProbePipelineConfig,
    subject: Subject,
    activation_grabber: ActivationGrabber,
    queries: List[str],
    probe_base_path: str | Path | None = None,
) -> dict | None:
    """
    Run (or ensure) a probe on provided queries. Returns probe_out or None.
    Queries are formatted with chat template to match training format.
    """
    base = probe_base_path or ensure_probe(cfg, subject, activation_grabber)
    if base is None:
        print("[Probe] No probe available; skipping inference.")
        return None
    # Format queries with chat template to match training format (BOS of response position)
    formatted_queries = [format_prompt_for_probe(q, subject.tokenizer, cfg.label_premise) for q in queries]
    probe_out = predict_with_probe(formatted_queries, subject, activation_grabber, str(base))
    for q, pred, prob_vec in zip(queries, probe_out["preds"], probe_out["probs"]):
        prob_correct = float(prob_vec[1]) if len(prob_vec) > 1 else float(prob_vec[0])
        print(f"[Probe] Q: {q}")
        print(f"  pred: {int(pred)}  prob(class=1): {prob_correct:.3f}")
    return probe_out


__all__ = [
    "DEFAULT_PROBE_DIR",
    "ProbeArtifacts",
    "TrainProbeResult",
    "build_probe_base_name",
    "extract_features",
    "load_labeled_json",
    "load_probe_artifacts",
    "predict_with_probe",
    "save_probe_artifacts",
    "train_probe",
    "load_nq_dataset",
    "get_model_response",
    "format_prompt_for_probe",
    "symmetric_verify_answer",
    "evaluate_model_on_dataset",
    "save_evaluation_results",
    "ProbePipelineConfig",
    "label_dataset_if_missing",
    "train_probe_if_missing",
    "ensure_probe",
    "run_probe_on_queries",
]
