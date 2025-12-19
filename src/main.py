from pathlib import Path

from hf_example_repo.activiation_grabber import ActivationGrabber
from hf_example_repo.subject import Subject, get_subject_config
from probe.hal_utils import (
    ProbePipelineConfig,
    ensure_probe,
    run_probe_on_queries,
)
from util.chat_input import ModelInput


def main() -> None:
    print("Hello World")
    # Use the same model for subject (activations) and labeling.
    config = get_subject_config("meta-llama/Llama-2-7b-chat-hf")
    subject = Subject(
        config=config,
        output_attentions=True,
        cast_to_hf_config_dtype=True,
        disable_flash_attention=True,
        nnsight_lm_kwargs={"dispatch": False}
    )
    activation_grabber = ActivationGrabber(subject)
    
    # Your query
    query = "What is the capital of France?"
    print(f"\nQuery: {query}")
    
    # 1. Generate response using chat template (empty system message)
    system_message = ""
    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": query}]
    formatted_prompt = subject.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ci = ModelInput(text=formatted_prompt)
    output = subject.generate(
        ci,
        max_new_tokens=10,
        temperature=1.0,
        stream=False
    )
    
    # 2. Get the generated response (strip EOS token)
    generated_text = output.output_strings[0].replace("</s>", "").strip()
    print(f"Response: {generated_text}")
    
    # --- Activation extraction demo (commented out, see README for usage) ---
    # full_sequence = query + generated_text
    # activation_data = activation_grabber.get_activations(full_sequence)
    # # activations shape: (batch, layers, tokens, hidden_dim)
    # # e.g., activation_data.activations[0, layer_idx, token_idx, :]

    # --- Optional: auto label/train/infer probe via hal_utils helpers ---
    PROBE_ENABLE = True  # set True to run pipeline
    cfg = ProbePipelineConfig(
        raw_dataset_path=Path("src/probe/data/nq_dataset.json"),
        labeled_path=Path("artifacts/labeled_nq.json"),
        probe_dir=Path("artifacts/probes"),
        label_model_id=config.hf_model_id,  # match subject model
        label_premise=None,
        label_max_samples=2000,  # training samples
        completeness=0.5,
        probe_layer=8, 
        probe_pos_mode="last",
        probe_pos_idx=None,
        probe_pool="none",
        probe_tag="base",
        label_key="is_correct",
        text_key="question",
        force_relabel=True,  # if any changes to labeling logic were made, set to True to re-label
    )

    if PROBE_ENABLE:
        base = ensure_probe(cfg, subject, activation_grabber)
        if base:
            # Probe is trained on questions only, so run on query (not query+response)
            run_probe_on_queries(cfg, subject, activation_grabber, [query], probe_base_path=base)
            print(f"Response: {generated_text}")


if __name__ == "__main__":
    main()
