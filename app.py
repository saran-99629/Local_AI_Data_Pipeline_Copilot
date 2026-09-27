import re
import requests
import streamlit as st

from rag_engine import retrieve_context


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"


# --------------------------------------------------
# 2. STREAMLIT PAGE
# --------------------------------------------------

st.set_page_config(
    page_title="Local AI Data Pipeline Copilot",
    page_icon="⚙️",
    layout="wide"
)

st.title("Local AI Data Pipeline Copilot")

st.caption(
    "Generate ETL workflows and PySpark code using "
    "a local LLM powered by RAG."
)

st.info(
    "Your requirement is matched against your local "
    "Data Engineering knowledge base before generation."
)


# --------------------------------------------------
# 3. USER INPUT
# --------------------------------------------------

requirement = st.text_area(
    "Describe your data pipeline requirement",
    placeholder=(
        "Example: Read customer data from CSV, "
        "remove duplicates, handle null values, "
        "and write the output to Parquet."
    ),
    height=150
)


# --------------------------------------------------
# 4. EXTRACT PYTHON CODE
# --------------------------------------------------

def extract_python_code(answer):
    """Extract Python code from a Markdown code block."""

    pattern = r"```(?:python|pyspark|py)\s*\n(.*?)```"

    match = re.search(
        pattern,
        answer,
        re.IGNORECASE | re.DOTALL
    )

    if match:
        return match.group(1).strip()

    return (
        "# No Python code block was detected.\n"
        "# Please review the Full Response tab."
    )


# --------------------------------------------------
# 5. RAG + LLM GENERATION
# --------------------------------------------------

def generate_pipeline(user_requirement):

    # Step 1: Retrieve relevant knowledge
    context, sources = retrieve_context(
        user_requirement,
        top_k=5
    )

    # Step 2: Build the augmented prompt
    prompt = f"""
You are an experienced Data Engineer.

Your task is to design a PySpark ETL pipeline.

USER REQUIREMENT:
{user_requirement}

RETRIEVED KNOWLEDGE:
{context}

INSTRUCTIONS:
1. Use the retrieved knowledge when relevant.
2. Do not invent facts or claim unsupported sources.
3. Clearly state assumptions about file paths and schemas.
4. Generate readable PySpark code with imports.
5. Include data quality checks.
6. Explain the transformations simply.
7. Do not execute the generated code.
8. If the retrieved knowledge is insufficient,
   use your general knowledge and state the assumptions.

Return the answer in these sections:

## 1. Pipeline Overview

## 2. ETL Steps

## 3. PySpark Code

Place the complete code inside a Python
Markdown code block.

## 4. Data Quality Checks

## 5. Explanation
"""

    # Step 3: Send the augmented prompt to Ollama
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=300
    )

    response.raise_for_status()

    answer = response.json().get("response", "")

    if not answer.strip():
        raise ValueError(
            "The model returned an empty response."
        )

    return answer, context, sources


# --------------------------------------------------
# 6. GENERATE PIPELINE
# --------------------------------------------------

if st.button("Generate Pipeline", type="primary"):

    if not requirement.strip():

        st.warning(
            "Please enter a data pipeline requirement."
        )

    else:

        try:

            with st.spinner(
                "Retrieving knowledge and generating your pipeline..."
            ):

                answer, context, sources = generate_pipeline(
                    requirement
                )

                code = extract_python_code(answer)

            st.success(
                "Your pipeline draft has been generated!"
            )

            # ------------------------------------------
            # 7. DISPLAY RESULTS
            # ------------------------------------------

            plan_tab, code_tab, rag_tab = st.tabs(
                [
                    "Pipeline Plan",
                    "PySpark Code",
                    "RAG Knowledge"
                ]
            )

            # Pipeline plan
            with plan_tab:

                st.markdown(answer)

                st.download_button(
                    label="Download Full Pipeline Notes",
                    data=answer,
                    file_name="pipeline_notes.md",
                    mime="text/markdown"
                )

            # Generated code
            with code_tab:

                st.subheader("Generated PySpark Code")

                st.code(
                    code,
                    language="python"
                )

                st.download_button(
                    label="Download PySpark Code",
                    data=code,
                    file_name="generated_pipeline.py",
                    mime="text/x-python"
                )

                st.warning(
                    "Review and test generated code before "
                    "running it on real data."
                )

            # Retrieved RAG context
            with rag_tab:

                st.subheader("Retrieved Knowledge")

                st.caption(
                    "These knowledge-base chunks were "
                    "provided to the LLM as context."
                )

                if context.strip():
                    st.markdown(context)
                else:
                    st.info(
                        "No relevant context was retrieved."
                    )

                st.subheader("Source Documents")

                if sources:
                    for source in sources:
                        st.write(f"- {source}")
                else:
                    st.write(
                        "No source documents were returned."
                    )

        except requests.exceptions.ConnectionError:

            st.error(
                "Cannot connect to Ollama. "
                "Make sure Ollama is running locally."
            )

        except requests.exceptions.Timeout:

            st.error(
                "The request timed out. "
                "Try a shorter requirement or try again."
            )

        except requests.exceptions.HTTPError as error:

            st.error(
                f"Ollama returned an HTTP error: {error}"
            )

        except requests.exceptions.RequestException as error:

            st.error(
                f"Could not communicate with Ollama: {error}"
            )

        except Exception as error:

            st.error(
                f"Something went wrong: {error}"
            )

            st.info(
                "Check that rag_engine.py, the knowledge_base "
                "folder, and ChromaDB are configured correctly."
            )


# --------------------------------------------------
# 8. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Built with Python, Streamlit, Ollama, Qwen, "
    "Ollama Embeddings, and ChromaDB."
)