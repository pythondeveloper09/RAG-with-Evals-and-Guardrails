import os
import time
import tempfile
import streamlit as st

# LangChain & Gemini Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Custom Modules
from evals import evaluate_rag_response
from guardrails import check_input_guardrails, check_output_guardrails

# Gemini 2.5 Flash Context Limit (1 Million Tokens)
GEMINI_FLASH_CONTEXT_WINDOW = 1_048_576

# Page Configuration
st.set_page_config(
    page_title="PDF QnA with Analytics & Guardrails",
    page_icon="🛡️",
    layout="wide"
)

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "current_file" not in st.session_state:
    st.session_state.current_file = None


def select_llm(model):
    if model == "Gemini":
        llm = ChatGoogleGenerativeAI(model='gemini-3.6-flash',google_api_key=api_key,temperature=0.2)
    elif model=="Groq":
        llm = ChatGroq(model='meta-llama/llama-prompt-guard-2-22m',api_key=api_key, temperature=0.2)
    else:
        llm = ChatOpenAI(model="gpt-4o-mini",api_key=api_key, temperature=0.2)
    return llm 


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def process_pdf(uploaded_file, api_key):
    """Saves PDF temporarily, splits into chunks, and creates FAISS index."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        split_docs = text_splitter.split_documents(documents)

        embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=api_key
        )

        vectorstore = FAISS.from_documents(split_docs, embeddings)
        return vectorstore
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)



def generate_rag_response(query, vectorstore, K, api_key):  # it returns 6 values
    """
    Executes RAG pipeline while tracking execution time, input/output tokens, 
    and remaining context window capacity.
    Return : 
    cleaned_response(str), list of page numbers(list), list of doc_sources(list of dict),
    eval_metrics(dict), input_violation_reason(str), perf_metrics(dict)

    """
    start_time = time.perf_counter()

    # 1. Input Guardrails
    input_pass, input_violation = check_input_guardrails(query)
    if not input_pass:
        latency = round(time.perf_counter() - start_time, 3)
        return None, [], [], None, input_violation, {"latency": latency, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "context_window": GEMINI_FLASH_CONTEXT_WINDOW, "remaining_context": GEMINI_FLASH_CONTEXT_WINDOW}

    # 2. Context Retrieval
    retriever = vectorstore.as_retriever(search_kwargs={"k": K})
    relevant_docs = retriever.invoke(query)

    context_text = ""
    pages_found = set()
    # subtitles_found = set()
    doc_sources = []

    for idx, doc in enumerate(relevant_docs):
        context_text += doc.page_content + "\n\n"
        page_num = doc.metadata.get("page", 0) + 1
        pages_found.add(page_num)
        
        doc_sources.append({
            "chunk_id": idx + 1,
            "page": page_num,
            "content": doc.page_content
        })

    # 3. Instantiate Gemini LLM
    llm = select_llm(model) 
    # ChatGoogleGenerativeAI(
    #     model="gemini-2.5-flash",
    #     google_api_key=api_key,
    #     temperature=0.2
    # )

    full_prompt = f"""You are an intelligent assistant trained to answer questions based strictly on the provided context.
If the answer cannot be determined from the context, state that the information is not in the document.

Context:
{context_text}

Question:
{query}

Answer:"""

    # 4. Generate Response
    response = llm.invoke(full_prompt)
    raw_answer = response.content[0]['text']

    latency = round(time.perf_counter() - start_time, 3)

    # 5. Calculate Token & Context Usage
    input_tokens = llm.get_num_tokens(full_prompt)
    output_tokens = llm.get_num_tokens(raw_answer)
    total_tokens = input_tokens + output_tokens
    remaining_context = max(0, GEMINI_FLASH_CONTEXT_WINDOW - total_tokens)

    # 6. Output Guardrails
    output_pass, output_violation, clean_answer = check_output_guardrails(
        query=query,
        context=context_text,
        response=raw_answer,
        api_key=api_key
    )

    # 7. Compute RAG Evaluation Metrics
    eval_metrics = evaluate_rag_response(
        query=query,
        context=context_text,
        response=clean_answer,
        api_key=api_key
    )

    # Calculate Total Latency 
    perf_metrics = {
        "latency": latency,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "context_window": GEMINI_FLASH_CONTEXT_WINDOW,
        "remaining_context": remaining_context
    }

    violation_reason = output_violation if not output_pass else ""

    return clean_answer, sorted(list(pages_found)),  doc_sources, eval_metrics, violation_reason, perf_metrics
    # list(subtitles_found),

def render_performance_and_token_metrics(perf: dict):
    """Displays latency, token usage breakdown, and remaining context capacity."""
    st.markdown("### ⚡ Execution Performance & Context Window")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(label="⏱️ Latency", value=f"{perf.get('latency', 0):.2f} s")

    with col2:
        st.metric(label="📥 Input Tokens", value=f"{perf.get('input_tokens', 0):,}")

    with col3:
        st.metric(label="📤 Output Tokens", value=f"{perf.get('output_tokens', 0):,}")

    with col4:
        st.metric(label="📊 Total Tokens", value=f"{perf.get('total_tokens', 0):,}")

    with col5:
        st.metric(label="📉 Remaining Context", value=f"{perf.get('remaining_context', 0):,}")

    st.caption(f"**Max Context Window Size:** {perf.get('context_window', 0):,} tokens")


def render_evaluation_dashboard(evals: dict):
    """Displays the LLM-as-a-Judge quality metrics."""
    st.markdown("### 📊 RAG Quality Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        f_score = evals.get("faithfulness", {}).get("score", 0.0)
        f_score_resoning = evals.get("faithfulness", {}).get("reasoning", "No reason generated")
        st.metric(label="Faithfulness", value=f"{f_score * 100:.0f}%")
        st.progress(min(max(f_score, 0.0), 1.0))
        st.caption(f_score_resoning)

    with col2:
        r_score = evals.get("recall", {}).get("score", 0.0)
        r_score_resoning = evals.get("recall", {}).get("reasoning", "No reason generated")
        st.metric(label="Context Recall", value=f"{r_score * 100:.0f}%")
        st.progress(min(max(r_score, 0.0), 1.0))
        st.caption(r_score_resoning)

    with col3:
        rel_score = evals.get("relevance", {}).get("score", 0.0)
        rel_score_reasoning = evals.get("relevance", {}).get("reasoning", "No reason generated")
        st.metric(label="Answer Relevance", value=f"{rel_score * 100:.0f}%")
        st.progress(min(max(rel_score, 0.0), 1.0))
        st.caption(rel_score_reasoning)

    with col4:
        c_score = evals.get("correctness", {}).get("score", 0.0)
        c_score_reasoning = evals.get("correctness", {}).get("reasoning", "No reason generated")
        st.metric(label="Correctness", value=f"{c_score * 100:.0f}%")
        st.progress(min(max(c_score, 0.0), 1.0))
        st.caption(c_score_reasoning)


# -----------------------------------------------------------------------------
# SIDEBAR UI
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    model = st.selectbox('Select LLM Provider',['Gemini','Groq','OpenAI'])
    if model == "Gemini":
        api_key = st.text_input("Enter Gemini API Key:", type="password")
    elif model=="Groq":
        api_key = st.text_input("Enter Groq API Key:", type="password")
    else :
        api_key = st.text_input("Enter OpenAI Key:", type="password")


    st.divider()
    st.header("📄 Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

    st.divider()
    K = st.slider('Retrieve K Chunks',3,10,5)
    st.caption(f'Top {K} chunks will be retieved')

    if uploaded_file and api_key:
        if st.session_state.current_file != uploaded_file.name:
            with st.spinner("Processing PDF and creating embeddings..."):
                try:
                    st.session_state.vectorstore = process_pdf(uploaded_file, api_key)
                    st.session_state.current_file = uploaded_file.name
                    st.session_state.messages = []
                    st.success(f"Loaded: **{uploaded_file.name}**")
                except Exception as e:
                    st.error(f"Error processing document: {e}")

    st.divider()
    if st.button("Start New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# -----------------------------------------------------------------------------
# MAIN CHAT INTERFACE
# -----------------------------------------------------------------------------

# st.session_state.messages = [{'role':'user','content':user_query}, 
#                              {'role':'assistant','content':response,  'pages': [],
#                              'guardrail_triggered': True/False,'guardrail_reason': True/False,
#                               'docs' : [{},{}],'perf_metrics':  {},'eval_metrics' : {} }]


st.title("🛡️ PDF QnA Chatbot with Guardrails & Performance Analytics")

if not api_key:
    st.warning("Please enter your Gemini API key in the sidebar to proceed.")
elif not uploaded_file:
    st.info("Please upload a PDF document in the sidebar to start asking questions.")
else:
    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            if msg["role"] == "assistant":
                if msg.get("guardrail_triggered"):
                    st.warning(f"⚠️ **Guardrail Warning:** {msg['guardrail_reason']}")

            
                if msg.get("pages"):
                    st.caption(f"📍 **Source Page(s):** Page {', '.join(map(str, msg['pages']))}")
                
                if msg.get("docs"):
                    with st.expander("🔍 View Retrieved Source Chunks"):
                        for doc in msg["docs"]:
                            st.markdown(f"**Chunk {doc['chunk_id']} (Page {doc['page']}) **")
                            st.text(doc["content"])
                            st.divider()

                if msg.get("perf_metrics"):
                    render_performance_and_token_metrics(msg["perf_metrics"])

                if msg.get("evals"):
                    render_evaluation_dashboard(msg["evals"])

    # User input chat box
    if user_query := st.chat_input("Ask a question about your document..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing request, enforcing guardrails & computing metrics..."):
                try:
                    answer, source_pages, doc_sources, eval_metrics, guardrail_reason, perf_metrics = generate_rag_response(
                        user_query,
                        st.session_state.vectorstore, K,
                        api_key
                    )

                    if answer is None:
                        blocked_msg = "I cannot answer this query as it violated safety guardrails."
                        st.markdown(blocked_msg)
                        st.error(f"⚠️ **Guardrail Warning:** {guardrail_reason}")
                        
                        if perf_metrics:
                            render_performance_and_token_metrics(perf_metrics)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": blocked_msg,
                            "guardrail_triggered": True,
                            "guardrail_reason": guardrail_reason,
                            "perf_metrics": perf_metrics
                        })
                    else:
                        st.markdown(answer)

                        if guardrail_reason:
                            st.warning(f"⚠️ **Guardrail Warning:** {guardrail_reason}")

                        if source_pages:
                            page_str = ", ".join(map(str, source_pages))
                            st.caption(f"📍 **Source Page(s):** Page {page_str}")

                        if doc_sources:
                            with st.expander("🔍 View Retrieved Source Chunks"):
                                for doc in doc_sources:
                                    st.markdown(f"**Chunk {doc['chunk_id']} (Page {doc['page']}) **")
                                    st.text(doc["content"])
                                    st.divider()

                        if perf_metrics:
                            render_performance_and_token_metrics(perf_metrics)

                        if eval_metrics:
                            render_evaluation_dashboard(eval_metrics)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "pages": source_pages,
                            "docs": doc_sources,
                            "evals": eval_metrics,
                            "perf_metrics": perf_metrics,
                            "guardrail_triggered": bool(guardrail_reason),
                            "guardrail_reason": guardrail_reason
                        })

                except Exception as e:
                    st.error(f"Failed to generate response: {e}")