import re
import json
from langchain_google_genai import ChatGoogleGenerativeAI

def evaluate_rag_response(query: str, context: str, response: str, api_key: str) -> dict:
    """Evaluates output across Faithfulness, Recall, Relevance, and Correctness."""
    eval_llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=api_key,
        temperature=0.0
    )

    eval_prompt = f"""You are an expert evaluator for Retrieval-Augmented Generation systems.
User Query: {query}
Retrieved Context: {context}
Generated Response: {response}

Evaluate on 4 metrics (0.0 to 1.0) and give 1-sentence reasoning for each.
Respond ONLY in JSON format:
{{
  "faithfulness": {{"score": 0.95, "reasoning": "Explanation"}},
  "recall": {{"score": 0.85, "reasoning": "Explanation"}},
  "relevance": {{"score": 0.90, "reasoning": "Explanation"}},
  "correctness": {{"score": 0.92, "reasoning": "Explanation"}}
}}
"""
    try:
        eval_result = eval_llm.invoke(eval_prompt).content[0]['text']
        clean_json = re.sub(r'```json\s*|\s*```', '', eval_result).strip()
        return json.loads(clean_json)
    except Exception as e:
        return {
            "faithfulness": {"score": 0.0, "reasoning": str(e)},
            "recall": {"score": 0.0, "reasoning": str(e)},
            "relevance": {"score": 0.0, "reasoning": str(e)},
            "correctness": {"score": 0.0, "reasoning": str(e)}
        }