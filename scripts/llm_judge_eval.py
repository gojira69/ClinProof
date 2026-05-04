import json
import os
import random
import time
from google import genai
from google.genai import types

# Set API key provided by user
os.environ["GEMINI_API_KEY"] = "AIzaSyDP5AkC1CmewMBv6u9bc-fEPavhwIF5Z3I"

def call_llm_judge(prompt: str) -> str:
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    model = "gemma-4-31b-it"
    
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level="HIGH",
        )
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=generate_content_config,
            )
            return response.text
        except Exception as e:
            err_str = str(e)
            if "429" in err_str and attempt < max_retries - 1:
                print(f"Rate limited (429). Waiting 45 seconds before retry...")
                time.sleep(45)
            else:
                print(f"Error calling Gemini: {e}")
                return ""
    return ""

def evaluate_sample(q_id, question, context, reasoning, propositions):
    prompt = f"""You are an expert clinical reasoning judge. Evaluate the following fact-checking trace.

QUESTION: {question}

ATOMIC PROPOSITIONS EXTRACTED:
{propositions if propositions else 'None'}

RETRIEVED CONTEXT:
{context if context else 'None'}

MODEL REASONING TRACE:
{reasoning}

Provide a strict evaluation using the following JSON format ONLY:
{{
    "M5_ADeQ_score": <int 0-10>, 
    "M7_Context_Utilization": <int 0 or 1>,
    "M8_Hallucination": <int 0 or 1>,
    "M9_Context_Relevancy": <int 0 or 1>
}}

Criteria:
- M5_ADeQ_score (0-10): How well do the atomic propositions break down the complex question into verifiable, granular sub-claims? (10=perfect coverage and atomicity, 0=terrible). If propositions are missing/None, score 0.
- M7_Context_Utilization (0 or 1): Score 1 if the reasoning explicitly relies on the provided RETRIEVED CONTEXT. Score 0 if it ignores the context.
- M8_Hallucination (0 or 1): Score 1 if the reasoning cites specific medical facts, statistics, or study names that are NOT present in the RETRIEVED CONTEXT. Score 0 if the reasoning is strictly faithful to the context (or its own parametric knowledge without inventing fake citations).
- M9_Context_Relevancy (0 or 1): Score 1 if the RETRIEVED CONTEXT contains information that is actually relevant and useful for answering the QUESTION. Score 0 if the context is irrelevant, unhelpful, or off-topic.
"""
    
    response = call_llm_judge(prompt)
    if not response:
        return {"M5_ADeQ_score": 0, "M7_Context_Utilization": 0, "M8_Hallucination": 0, "M9_Context_Relevancy": 0}
        
    try:
        # Try direct parse
        return json.loads(response)
    except Exception:
        try:
            # Try finding a JSON block
            start = response.find("{")
            end = response.rfind("}")
            if start != -1 and end != -1:
                return json.loads(response[start:end+1])
        except Exception as e:
            print(f"JSON parsing error: {e}")
            
    return {"M5_ADeQ_score": 0, "M7_Context_Utilization": 0, "M8_Hallucination": 0, "M9_Context_Relevancy": 0}

def main():
    print(f"=== ClinProof LLM-as-Judge Evaluation (M5, M7, M8, M9) ===")
    print(f"Judge Model: Gemini-3.1-Pro-Preview (Thinking: High)\n")
    
    targets = {
        "BioASQ": "results/B3_dense_bm25_bioasq.json",
        "HealthFC": "results/B4_full_pipeline_healthfc_test.json"
    }
    
    sample_size = 10  # N to evaluate per dataset
    
    for name, path in targets.items():
        path = path.replace("/", os.sep)
        if not os.path.exists(path):
            print(f"Skipping {name}, file not found: {path}")
            continue
            
        with open(path, 'r', encoding='utf-8') as f:
            results = json.load(f).get("results", [])
            
        # Filter to those with reasoning traces
        results = [r for r in results if r.get("reasoning_traces")]
        
        if not results:
            print(f"[{name}] No reasoning traces found.")
            continue
            
        # Sample N
        samples = random.sample(results, min(sample_size, len(results)))
        
        print(f"Evaluating {len(samples)} samples for {name}...")
        
        m5_scores = []
        m7_scores = []
        m8_scores = []
        m9_scores = []
        
        for i, r in enumerate(samples):
            print(f"  Grading {i+1}/{len(samples)}...", end="\r")
            
            trace = r["reasoning_traces"][0]
            reasoning = trace.get("step_by_step_thinking", "")
            context = r.get("retrieved_context", "")
            
            # Extract atomic propositions from the top of the context if present
            props = ""
            if context and "Key medical claims to verify:" in context:
                parts = context.split("\n\n")
                props = parts[0]
            
            eval_res = evaluate_sample(r.get("id"), r.get("question"), context, reasoning, props)
            
            m5_scores.append(eval_res.get("M5_ADeQ_score", 0))
            m7_scores.append(eval_res.get("M7_Context_Utilization", 0))
            m8_scores.append(eval_res.get("M8_Hallucination", 0))
            m9_scores.append(eval_res.get("M9_Context_Relevancy", 0))
            
            # Rate limiting for Gemini API to be safe
            time.sleep(2)
            
        print(f"\n[{name}] Judge Results (n={len(samples)}):")
        print(f"  M5: ADeQ Score        : {sum(m5_scores)/len(m5_scores):.1f}/10.0 (Atomic Decomp Quality)")
        print(f"  M7: Context Util      : {(sum(m7_scores)/len(m7_scores))*100:.1f}% (Used retrieved context)")
        print(f"  M8: Hallucination Rate: {(sum(m8_scores)/len(m8_scores))*100:.1f}% (Cited facts NOT in context)")
        print(f"  M9: Context Relevancy : {(sum(m9_scores)/len(m9_scores))*100:.1f}% (Context was relevant to query)")
        print("-" * 50)

if __name__ == "__main__":
    main()
