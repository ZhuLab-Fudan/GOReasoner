"""
GOExplainer: LLM-based GO term verification model.
Handles model inference and confidence scoring.
"""

import re
from typing import Dict, List, Tuple, Optional
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

from .scoring import GOConfidenceScorer
from .prompt_template import PromptTemplate


class GOExplainer:
    """
    Large Language Model for GO term verification and scoring.
    """
    
    def __init__(self,
                 model_path: str,
                 lora_path: Optional[str] = None,
                 max_model_len: int = 8192,
                 gpu_memory_utilization: float = 0.8,
                 temperature: float = 0.0,
                 max_tokens: int = 256):
        """
        Initialize GOExplainer model.
        
        Args:
            model_path: Path to base LLM model
            lora_path: Optional path to LoRA adapter
            max_model_len: Maximum sequence length
            gpu_memory_utilization: GPU memory usage ratio
            temperature: Sampling temperature
            max_tokens: Maximum generation tokens
        """
        self.model_path = model_path
        self.lora_path = lora_path
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Initialize vLLM model
        self.llm = LLM(
            model=model_path,
            dtype="bfloat16",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_lora=True,
            enable_prefix_caching=True
        )
        
        # Sampling parameters
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop_token_ids=[self.tokenizer.eos_token_id],
            logprobs=1
        )
        
        # LoRA request
        self.lora_request = None
        if lora_path:
            self.lora_request = LoRARequest("gollm_adapter", 1, lora_path)
        
        # Initialize scorer and prompt template
        self.scorer = GOConfidenceScorer()
        self.prompt_template = PromptTemplate()
        
        # Cache for log probabilities
        self.log_probs_cache = {}
    
    def _parse_answer(self, 
                     answer: str, 
                     goid: str) -> Tuple[bool, str]:
        """
        Parse model answer and extract True/False judgement.
        
        Args:
            answer: Model generated answer
            goid: GO term ID
            
        Returns:
            Tuple of (is_true, parsed_answer)
        """
        try:
            if len(answer.split()) < 4:
                if 'True' in answer:
                    return True, f"{goid}\tTrue\t{answer}"
                elif 'False' in answer:
                    return False, f"{goid}\tFalse\t{answer}"
                else:
                    return False, f"{goid}\tFalse\tParse Error: {answer}"
            
            judgement = answer.split()[1]
            if judgement == 'True':
                return True, answer
            elif judgement == 'False':
                return False, answer
            else:
                # Fallback to string search
                if 'True' in answer:
                    return True, answer
                elif 'False' in answer:
                    return False, answer
                else:
                    return False, f"{goid}\tFalse\tParse Error: {answer}"
        except Exception as e:
            return False, f"{goid}\tFalse\t{str(e)}{answer}"
    
    def _truncate_statement(self,
                           statement: str,
                           protein_name: str,
                           domain: str,
                           max_prefix_len: int = 7680) -> str:
        """
        Truncate statement text to fit within token limit.
        
        Args:
            statement: Full statement text
            protein_name: Protein name
            domain: GO domain
            max_prefix_len: Maximum prefix length
            
        Returns:
            Truncated statement text
        """
        # Calculate base prompt length
        temp_input = self.prompt_template.format_input(
            protein_name=protein_name,
            go_domain=domain,
            statement_text="",
            candidates=""
        )
        temp_messages = self.prompt_template.create_messages(temp_input)
        prompt_base = self.tokenizer.apply_chat_template(
            temp_messages, tokenize=False, 
            add_generation_prompt=True, enable_thinking=False
        )
        
        len_base = len(self.tokenizer.encode(prompt_base))
        available_len = max(0, max_prefix_len - len_base)
        
        # Truncate statement if necessary
        statement_tokens = self.tokenizer.encode(statement)
        if len(statement_tokens) > available_len:
            truncated_tokens = statement_tokens[:available_len]
            return self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        
        return statement
    
    def get_score(self,
                 protein_id: str,
                 statement: str,
                 domain: str,
                 go_list: List[str],
                 go_definitions: Dict[str, str],
                 batch_size: int = 32) -> Tuple[Dict[str, float], Dict[str, str]]:
        """
        Calculate confidence scores for GO term candidates.
        
        Args:
            protein_id: Protein identifier
            statement: Protein description
            domain: GO domain (mf/bp/cc)
            go_list: List of candidate GO terms
            go_definitions: Mapping of GO IDs to definitions
            batch_size: Batch size for inference
            
        Returns:
            Tuple of (score_dict, reason_dict)
        """
        score_dict = {}
        reason_dict = {}
        
        # Truncate statement
        protein_name = protein_id  # Simplified, should load from proid2name
        truncated_statement = self._truncate_statement(
            statement, protein_name, domain
        )
        
        # Process in batches
        go_batches = [
            go_list[i:i + batch_size] 
            for i in range(0, len(go_list), batch_size)
        ]
        
        for batch in go_batches:
            batch_prompts = []
            uncached_goids = []
            
            for goid in batch:
                # Check cache
                if self._is_cached(protein_id, goid):
                    cached_score, cached_reason = self._get_cached_result(
                        protein_id, goid
                    )
                    score_dict[goid] = cached_score
                    reason_dict[goid] = cached_reason
                    continue
                
                # Prepare prompt
                input_candidate = f"**{goid}**: {go_definitions[goid]}"
                formatted_input = self.prompt_template.format_input(
                    protein_name=protein_name,
                    go_domain=domain,
                    statement_text=truncated_statement,
                    candidates=input_candidate
                )
                messages = self.prompt_template.create_messages(formatted_input)
                
                batch_prompts.append(messages)
                uncached_goids.append(goid)
            
            if not batch_prompts:
                continue
            
            # Batch inference with confidence scoring
            outputs = self.scorer.get_generation_segment_confidence_batch(
                llm=self.llm,
                tokenizer=self.tokenizer,
                batch_messages=batch_prompts,
                sampling_params=self.sampling_params,
                lora_request=self.lora_request
            )
            
            # Process outputs
            for i, output in enumerate(outputs):
                goid = uncached_goids[i]
                answer, base_score, logprobs, token_ids, tokens = output
                
                is_true, parsed_answer = self._parse_answer(answer, goid)
                final_score = base_score if is_true else -1 * base_score
                
                score_dict[goid] = final_score
                reason_dict[goid] = parsed_answer
                
                # Cache result
                self._cache_result(protein_id, goid, logprobs, token_ids)
        
        return score_dict, reason_dict
    
    def _is_cached(self, protein_id: str, goid: str) -> bool:
        """Check if result is cached."""
        if protein_id not in self.log_probs_cache:
            return False
        if goid not in self.log_probs_cache[protein_id]:
            return False
        
        cached_data = self.log_probs_cache[protein_id][goid]
        decoded = self.tokenizer.decode(cached_data[-1])
        
        return len(decoded.split()) > 2 and goid in decoded.split()
    
    def _get_cached_result(self, 
                          protein_id: str, 
                          goid: str) -> Tuple[float, str]:
        """Retrieve cached scoring result."""
        cached_logprobs = self.log_probs_cache[protein_id][goid]
        
        # Calculate score
        tokens = [
            self.tokenizer.decode(tid, skip_special_tokens=True) 
            for tid in cached_logprobs[-1]
        ]
        score = self.scorer._calculate_segmented_confidence(
            tokens=tokens,
            log_token_probs=cached_logprobs[0]
        )
        
        # Parse answer
        answer = self.tokenizer.decode(cached_logprobs[-1], skip_special_tokens=True)
        is_true = answer.split()[1] == 'True' if len(answer.split()) > 1 else 'True' in answer
        
        if not is_true:
            score = -1 * score
        
        return score, answer
    
    def _cache_result(self, 
                     protein_id: str, 
                     goid: str, 
                     logprobs, 
                     token_ids):
        """Cache inference results."""
        if protein_id not in self.log_probs_cache:
            self.log_probs_cache[protein_id] = {}
        
        self.log_probs_cache[protein_id][goid] = (logprobs, None, token_ids)
    
    def save_cache(self, cache_path: str):
        """Save log probability cache to disk."""
        import numpy as np
        np.save(cache_path, self.log_probs_cache)
    
    def load_cache(self, cache_path: str):
        """Load log probability cache from disk."""
        import numpy as np
        import os
        
        if os.path.exists(cache_path):
            self.log_probs_cache = np.load(cache_path, allow_pickle=True).item()
            print(f"Loaded cache from {cache_path}")