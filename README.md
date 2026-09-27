# Local AI Data Pipeline Copilot

A local LLM-powered assistant that helps Data Engineers
draft ETL workflows, PySpark code, and data quality checks.

## Features

- Generate ETL pipeline plans from natural language
- Generate PySpark code
- Suggest data quality validations
- Explain pipeline transformations
- Download generated code and documentation
- Run inference locally using Ollama

## Tech Stack

- Python
- Ollama
- Qwen 2.5 3B
- Streamlit
- Requests

## Architecture

User Requirement
      |
      v
Streamlit UI
      |
      v
Python Prompt Builder
      |
      v
Ollama Local API
      |
      v
Qwen 2.5 3B
      |
      v
Pipeline Plan + PySpark Code

## Run Locally

1. Install Ollama.
2. Download the model:

   ollama run qwen2.5:3b

3. Install dependencies:

   python -m pip install requests streamlit

4. Start the application:

   python -m streamlit run app.py

## Important

Generated code is provided as a draft.
Review and test it before executing it.
