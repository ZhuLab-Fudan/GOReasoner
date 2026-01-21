import math
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from typing import Optional, List, Dict, Tuple

def _calculate_confidence(
    tokens: List[str],
    log_token_probs: List[float],
    min_k_ratio: float = 0.2,      # Ratio for Min-K% (e.g., bottom 20% least probable tokens)
    alpha: float = 0.7,           # Weight for Judgement score (Equation 2/3 alignment)
    exclude_punct: bool = True     # Filter noise characters
) -> float:
    """
    Calculates the Judgement-Reason Joint (JRJ) Confidence Score:
    - Judgement (Pj): Extracted from the binary decision token.
    - Reason (Pr): Calculated using Min-K% (geometric mean of least probable tokens).
    - Combination: Pj^alpha * Pr^(1-alpha).
    """
    n = len(log_token_probs)
    if n == 0: return 0.0

    # Convert log-probabilities to linear probability space
    probs = [math.exp(lp) for lp in log_token_probs]

    # Semantic segmentation based on tab separators [cite: 209]
    # Structure: [0:GOID, 1:Judgement, 2:Evidence, 3:Reason]
    sections_data = [[] for _ in range(4)]
    current_sec = 0
    
    noise_chars = {'"', "'", "“", "”", "<|im_end|>", ".", ",", " "}

    for i, token in enumerate(tokens):
        if "\t" in token:
            if current_sec < 3: current_sec += 1
            continue
        
        # Filter noise and punctuation to focus on content-rich tokens
        clean_token = token.strip()
        if exclude_punct and (not clean_token or clean_token in noise_chars):
            continue
            
        sections_data[current_sec].append(probs[i])

    # --- 1. Calculate Judgement Score (Pj) ---
    # Usually a single token (True/False). We take the minimum probability in the segment.
    p_j = min(sections_data[1]) if sections_data[1] else 1e-5

    # --- 2. Calculate Reason Score (Pr) using Min-K% ---
    reason_probs = sections_data[3]
    if not reason_probs:
        p_r = 1e-5
    else:
        # Sort probabilities to find the k% least-probable tokens [cite: 134, 135]
        reason_probs.sort()
        num_k = max(1, int(len(reason_probs) * min_k_ratio))
        mink_tokens = reason_probs[:num_k]
        
        # Use geometric mean of Min-K tokens 
        log_mink_sum = sum(math.log(max(p, 1e-10)) for p in mink_tokens)
        p_r = math.exp(log_mink_sum / num_k)

    # --- 3. Final Consensus Score (Geometric Combination) ---
    # As per paper's integration of uncertain points [cite: 138]
    final_confidence = (p_j ** alpha) * (p_r ** (1 - alpha))
    
    return final_confidence

def get_generation_confidence_batch(
    llm: LLM,
    tokenizer: AutoTokenizer,
    batch_messages: List[List[Dict[str, str]]],
    min_k_ratio: float = 0.2,
    alpha: float = 0.7,
    sampling_params: SamplingParams = SamplingParams(),
    lora_request: Optional[any] = None
) -> List[Tuple[str, float, List[float], List[int]]]:
    """
    Batch processing version of GOReasoner confidence extraction.
    Returns a list of: (generated_text, confidence_score, raw_logprobs, token_ids)
    """
    
    # Ensure logprobs are requested from vLLM
    if sampling_params.logprobs is None:
        sampling_params.logprobs = 1

    # Prepare prompts using chat template
    prompt_strs = tokenizer.apply_chat_template(
        batch_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Batch generation
    if lora_request is not None:
        batch_outputs = llm.generate(prompt_strs, sampling_params, lora_request=lora_request, use_tqdm=False)
    else:   
        batch_outputs = llm.generate(prompt_strs, sampling_params, use_tqdm=False)
    
    results = []
    for output in batch_outputs:
        if not output.outputs:
            results.append(("Generation Error", 0.0, [], []))
            continue

        gen_output = output.outputs[0]
        generated_text = gen_output.text
        generated_token_ids = gen_output.token_ids
        logprobs_dicts_list = gen_output.logprobs

        if logprobs_dicts_list is None or not generated_token_ids:
            results.append((generated_text, 0.0, [], []))
            continue

        # Extract log-probabilities for the chosen tokens
        try:
            raw_logprobs = [
                logprobs_dicts_list[i][tid].logprob 
                for i, tid in enumerate(generated_token_ids)
            ]
        except (KeyError, IndexError):
            results.append((generated_text, 0.0, [], []))
            continue

        # Calculate JRJ Confidence Score
        final_score = _calculate_confidence(
            tokens=[tokenizer.decode(tid) for tid in generated_token_ids],
            log_token_probs=raw_logprobs,
            min_k_ratio=min_k_ratio,
            alpha=alpha
        )
        final_score = (final_score + 1) / 2 # Normalize to [0, 1]
        results.append((generated_text, final_score, raw_logprobs, generated_token_ids))
        
    return results