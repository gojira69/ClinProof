"""
ClinProof Ollama LLM Wrapper
LangChain-compatible Ollama client for fast local inference.
Supports: qwen2.5:14b, qwen2.5:7b, mistral:7b, llama3.2:3b, phi4:14b
"""
import logging
import re
import json
from ollama import Client as OllamaClient

log = logging.getLogger("ollama_llm")

# Context lengths per model tag
MODEL_CONTEXT_LENGTHS = {
    "qwen2.5:14b":    32768,
    "qwen2.5:7b":     32768,
    "qwen2.5:3b":     32768,
    "mistral:7b":     32768,
    "llama3.2:3b":    128000,
    "llama3.2:1b":    128000,
    "phi4:14b":       16384,
    "phi4-mini:3.8b": 16384,
    "medllama2:7b":   4096,
}


class OllamaLLM:
    """ClinProof LLM interface backed by Ollama."""

    def __init__(self, config: dict):
        model_cfg = config.get("model", {})
        self.model_name = model_cfg.get("name", "qwen2.5:7b")
        self.temperature = model_cfg.get("temperature", 0)
        self.max_new_tokens = model_cfg.get("max_new_tokens", 1024)
        self.context_length = model_cfg.get(
            "context_length",
            MODEL_CONTEXT_LENGTHS.get(self.model_name, 4096)
        )
        self.client = OllamaClient()
        log.info(f"OllamaLLM init: model={self.model_name}, ctx={self.context_length}")

    def generate(self, messages: list, **kwargs) -> str:
        try:
            opts = {
                "temperature": self.temperature,
                "num_predict": self.max_new_tokens,
            }
            opts.update(kwargs)  # allow caller to override temperature etc.
            resp = self.client.chat(
                model=self.model_name,
                messages=messages,
                options=opts
            )
            return resp["message"]["content"].strip()
        except Exception as e:
            log.error(f"Ollama generation error: {e}")
            return ""

    def extract_json(self, text: str) -> dict:
        try:
            return json.loads(text)
        except Exception:
            pass
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return {}

    def answer_mcq(self, question: str, options: dict, context: str = "", system_prompt: str = None, votes: int = 3) -> dict:
        """Answer an MCQ with self-consistency voting + double-pass verification."""
        if not context or votes <= 1:
            return self._single_pass(question, options, context, system_prompt)

        # Self-consistency: sample `votes` times, take majority
        from collections import Counter
        candidates = []
        for _ in range(votes):
            r = self._single_pass(question, options, context, system_prompt, temperature=0.3)
            candidates.append(r.get("answer_choice", "A"))

        vote_counts = Counter(candidates)
        best_choice, _ = vote_counts.most_common(1)[0]

        # Return the full result from the run that produced the winning answer
        for _ in range(votes):
            r = self._single_pass(question, options, context, system_prompt, temperature=0)
            if r.get("answer_choice") == best_choice:
                return r

        return self._single_pass(question, options, context, system_prompt)

    def _single_pass(self, question: str, options: dict, context: str = "", system_prompt: str = None, temperature: int = None) -> dict:
        """Single MCQ attempt."""
        options_str = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
        if context:
            user_content = (
                f"Relevant documents:\n{context}\n\n"
                f"Question: {question}\n\nOptions:\n{options_str}\n\n"
                'Respond with JSON only: {"step_by_step_thinking": "...", "answer_choice": "A"}'
            )
        else:
            user_content = (
                f"Question: {question}\n\nOptions:\n{options_str}\n\n"
                'Respond with JSON only: {"step_by_step_thinking": "...", "answer_choice": "A"}'
            )
        if system_prompt is None:
            system_prompt = (
                "You are a medical expert. Answer the question based on provided evidence. "
                "Always respond with valid JSON only."
            )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        # Use caller-specified temp or default to 0
        gen_kwargs = {}
        if temperature is not None:
            gen_kwargs["temperature"] = temperature

        raw_1 = self.generate(messages, **gen_kwargs)
        parsed_1 = self.extract_json(raw_1)
        if "answer_choice" not in parsed_1:
            match = re.search(r'\b([ABCDE])\b', raw_1)
            parsed_1["answer_choice"] = match.group(1) if match else "A"
            
        if "step_by_step_thinking" not in parsed_1:
            parsed_1["step_by_step_thinking"] = raw_1
            
        return parsed_1

    def answer_open(self, question: str, context: str = "", system_prompt: str = None) -> str:
        """Answer an open-ended question (PubMedQA, BioASQ yes/no)."""
        if context:
            user_content = f"Documents:\n{context}\n\nQuestion: {question}\n\nAnswer concisely:"
        else:
            user_content = f"Question: {question}\n\nAnswer:"
        if system_prompt is None:
            system_prompt = "You are a medical expert. Answer based on evidence."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        return self.generate(messages)
