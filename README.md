# 🛡️ PDF QnA Chatbot with Guardrails & Performance Analytics

A Retrieval-Augmented Generation (RAG) Streamlit application that lets users chat with PDF documents while enforcing multi-layer safety guardrails, evaluating response quality via LLM-as-a-Judge, and tracking real-time token and latency metrics.

## ✨ Features

* **Multi-LLM Support:** Switch between **Gemini**, **Groq**, and **OpenAI** providers directly from the sidebar.
* **Interactive PDF RAG Pipeline:** Splits uploaded PDFs into chunks (`chunk_size=1000`, `chunk_overlap=50`), generates embeddings with `gemini-embedding-001`, and indexes them locally in **FAISS**.
* **Configurable Retrieval & Citations:** Adjust top-$K$ chunk retrieval (`3` to `10`) and inspect exact source page numbers and retrieved chunk text for every answer.
* **Input & Output Guardrails:**
  * **Input Protection:** Blocks prompt injection and jailbreak attempts before invoking the RAG pipeline.
  * **PII Redaction:** Automatically detects and masks sensitive data (Credit Card numbers, SSNs, IBAN/Bank Accounts, and API keys) in model responses.
  * **Safety & Hallucination Checks:** Evaluates generated responses to flag hallucinations and toxic content.
* **LLM-as-a-Judge Evaluation Dashboard:** Scores every response (`0%`–`100%`) with 1-sentence explanations across four RAG metrics:
  * **Faithfulness**
  * **Context Recall**
  * **Answer Relevance**
  * **Correctness**
* **Performance & Token Analytics:** Tracks end-to-end latency, input/output/total token counts, and remaining context window capacity.

## 📂 Project Structure

```text
├── app.py              # Main Streamlit UI, FAISS indexing, RAG execution, and metrics rendering
├── guardrails.py       # Input prompt-injection checks, PII regex masking, and output safety checks
├── evals.py            # LLM-as-a-Judge RAG evaluation (Faithfulness, Recall, Relevance, Correctness)
└── requirements.txt    # Python dependencies
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run app.py
```

## 🛠️ Usage

1. Open the local URL displayed by Streamlit in your browser.
2. Select an **LLM Provider** (`Gemini`, `Groq`, or `OpenAI`) in the sidebar and enter your API key. *(Note: Document embeddings, output guardrails, and RAG evaluations use Google Gemini models and require a valid Gemini API key.)*
3. Upload a **PDF document** and set the **Retrieve K Chunks** slider.
4. Ask questions in the chat input to view grounded answers, source page citations, guardrail warnings, latency/token metrics, and RAG quality scores.

## 📦 Tech Stack

* **Frontend:** Streamlit
* **Orchestration:** LangChain (`langchain`, `langchain-core`, `langchain-community`, `langchain-text-splitters`)
* **Vector Store:** FAISS (`faiss-cpu`)
* **LLM & Embedding Integrations:** `langchain-google-genai`, `langchain-groq`, `langchain-openai`
* **PDF Parsing:** PyPDF
````[cite: 1, 2, 3, 4]
