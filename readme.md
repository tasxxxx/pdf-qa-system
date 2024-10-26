# PDF Question Answering System
**Author:** Babak Hosseini

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [System Architecture](#system-architecture)
   - [Key Components in the Codebase](#key-components-in-the-codebase)
- [Design Decisions and Technology Choices](#design-decisions-and-technology-choices)
   - [Hybrid Retrieval Strategy](#hybrid-retrieval-strategy)
   - [Query Shortening](#query-shortening)
   - [Cross-Encoder Scoring](#cross-encoder-scoring)
   - [LLM Instruction for Accuracy](#llm-instruction-for-accuracy)
- [Assumptions](#assumptions)
- [Installation](#installation)
   - [Prerequisites](#prerequisites)
   - [Setup Instructions \[no docker\]](#setup-instructions-no-docker)
   - [Setup Instructions using Docker:](#setup-instructions-using-docker)
- [Usage](#usage)
   - [Running the Application](#running-the-application)
   - [Using the Application:](#using-the-application)
   - [Parameter Descriptions](#parameter-descriptions)
- [External Services](#external-services)
- [Further Enhancements](#further-enhancements)

---

## Introduction

This application enables users to upload a collection of PDF documents and interactively ask questions related to their content. Using advanced large language model (LLM) toolchains and techniques, the system processes the documents, extracts information relevant to the user's question, and provides meaningful responses based on the facts in the provided documents.

## Project Structure
Below is an overview of the project's structure:

```plaintext
pdf-qa-system/
├── src/
│   ├── app.py
│   └── helper_functions.py
├── configs/
│   └── config.yaml
├── data/
│   └── pdfs_files/
├── requirements.txt
├── Dockerfile (optional)
├── README.md
├── .env
└── .gitignore
```

## System Architecture

The following describes the flow of data within the PDF Question Answering System:

1. **PDF Ingestion:** Users upload PDF documents, which are loaded and split into manageable chunks.
2. **Vector Store Creation:** Chunks are embedded and stored in ChromaDB for efficient retrieval.
3. **Question Processing:** User queries are shortened and used to retrieve relevant document chunks via BM25 and semantic search.
4. **Answer Generation:** Retrieved chunks are ranked, and the top results are used by the LLM to generate accurate answers.

### Key Components in the Codebase

- **PDFManager:** 
  - **Responsibilities:** Loading PDFs, splitting documents into small and large chunks, and creating the vector store.
  - **Technologies Used:** PyPDFLoader, RecursiveCharacterTextSplitter, HuggingFaceEmbeddings, ChromaDB.

- **Retrievers:** 
  - **Responsibilities:** Implementing the hybrid retrieval strategy using BM25 and semantic search, setting up cross-encoder reranking.
  - **Technologies Used:** BM25Retriever, CrossEncoderReranker, ContextualCompressionRetriever, SelfQueryRetriever.

- **QAchains:** 
  - **Responsibilities:** Managing the question-answering pipeline, including query shortening, chunk retrieval, ranking, and answer generation.
  - **Technologies Used:** LLMChain, PromptTemplate, CrossEncoder, ChatOpenAI.


## Design Decisions and Technology Choices
### Hybrid Retrieval Strategy
To ensure both precision and recall in retrieving relevant information from the uploaded PDFs, a hybrid retrieval approach was implemented. This strategy combines keyword-based search with semantic search:

1- **Keyword Search using BM25:**
- Method: Utilized the BM25 algorithm to perform keyword-based retrieval.
- Purpose: Quickly identifies documents that contain keywords relevant to the user's query.
- Benefit: Efficiently narrows down the pool of documents to those most likely containing pertinent information.

2- **Semantic Search on Selected Documents:**
- Method: Applied semantic search within the subset of documents identified by the BM25 retriever.
- Purpose: Discovers semantically relevant passages that may not contain exact keyword matches but are contextually related.
- Benefit: Enhances the relevance of retrieved information by understanding the contextual meaning of the query.

3- **Meta Information and Self Query Retrieval:**
- Method: Leveraged meta information (e.g., document name) to constrain the semantic search to specific documents.
- Purpose: Ensures that semantic search is performed only within the documents deemed relevant by the keyword search.
- Benefit: Improves retrieval accuracy by focusing semantic analysis on a targeted subset of documents.

### Query Shortening
To optimize the retrieval process, especially for complex or lengthy user queries, a query shortening mechanism was implemented:

- Functionality: Reduces the original, potentially verbose query to a concise set of essential keywords.
- Method: Utilized an LLM to extract key phrases and terms that capture the core intent of the query.
- Purpose: Enhances the effectiveness of the BM25 keyword search by focusing on the most relevant terms.
- Benefit: Improves retrieval precision by eliminating noise and irrelevant terms from the query.

### Cross-Encoder Scoring
To further refine the relevance of retrieved documents and passages, cross-encoder models were employed for scoring:

- Functionality: Assigns relevance scores to both keyword-based and semantically retrieved documents.
- Method: Utilized cross-encoder models to evaluate the similarity between the user's query and the content of each document chunk.
- Purpose: Provides a more nuanced and accurate ranking of documents based on their contextual relevance to the query.
- Benefit: Enhances the quality of the final retrieved set by prioritizing the most contextually appropriate documents.

### LLM Instruction for Accuracy
To ensure the generated answers are both accurate and grounded in the provided documents, specific instructions were embedded within the LLM's prompt:

- Instruction: The LLM is instructed to limit its responses strictly to the knowledge extracted from the uploaded documents and to refrain from generating information beyond what is supported by the text.
- Mechanism: The prompt explicitly directs the LLM to avoid hallucinations and to acknowledge when information is not available.
- Purpose: Maintains the reliability and factual accuracy of the responses.
- Benefit: Prevents the model from fabricating answers, ensuring that users receive trustworthy information based solely on their provided documents.

---

## Assumptions
While building this application, the following assumptions were made:

1- **Document Language:**
All PDF documents are written in English.

2- **PDF Content:**
PDFs primarily contain textual data. Images, tables, or scanned documents requiring OCR are not handled in the current implementation. 

3- **Document Size**:
Each PDF is of reasonable size and does not exceed typical page limits (e.g., 100 pages).

4- **User Environment:**
Users have access to a compatible **Python environment V3.12** and can install necessary dependencies.

5- **API Availability:**
External services like OpenAI's GPT models and HuggingFace's embedding models are accessible and operational during usage.

6- **Content Specificity:**
The questions and documents are specific to particular entities such as companies, consultancy services, or stocks, rather than being generic. This specificity justifies the use of keyword-based retrieval in conjunction with semantic search to accurately identify relevant information.

7- **Use Case Relevance:**
The current design is optimized for scenarios where users inquire about specific, non-generic topics related to the uploaded documents. Different use cases with more generic or diverse queries might necessitate alternative design approaches.

---

## Installation

### Prerequisites

Before setting up the application, ensure you have the following installed on your system:

- **Python:** Version 3.12 or higher
- **pip:** Python package installer
- **Git:** For cloning the repository
- **Docker:** [optional] For a more convenient setup and deployment

### Setup Instructions [no docker]

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/pdf-qa-system.git
   cd pdf-qa-system
   ```   
2. **Create a Virtual Environment**
   It is recommended to create a virtual environment to isolate the dependencies.
   ```bash
   python -m venv .venv
   source .venv/bin/activate  [for linux or mac]
   ```
3. **Install Dependencies:**
   Dependencies are installed using `pip`.
   ```bash
   pip install -r requirements.txt
   ```

4. **Configuration:**
   The `config.yaml` contains default settings for the application. 
   You do not need to change them for the initial demo run. 
   But you can modify these settings to customize the behavior of the application.
   The application uses OpenAI's `GPT-4o` model for LLM calls. 

5. **.env file**
   To use OpenAI model, you need to set the `OPENAI_API_KEY` environment variable.
   Set it appropriately in `.env_example` and copy it as `.env`.   
   **Optional:** You can also set the `LANGSMITH_API_KEY` environment variable if you want to use LangSmith for tracing.

### Setup Instructions using Docker:
To set up the application using Docker, skip steps 2 and 3 from the above, perform steps 4 and 5 and then run the following command:

``` bash
docker build -t pdf-qa-system .
```

---

## Usage
### Running the Application
 The application uses streamlit for building the user interface.
 - **No Docker Setup:**
 ```bash 
 streamlit run app.py
 ```
 - **With Docker Setup:**

for windows use:
``` bash
docker run -p 8501:8501 --env-file .env -v ${PWD}/data:/app/data pdf-qa-system
```
for Linux use:
``` bash
docker run -p 8501:8501 --env-file .env -v $(pwd)/data:/app/data pdf-qa-system
```
Open your browser and navigate to the URL provided in the terminal (usually http://localhost:8501).

### Using the Application:

1. **Upload PDF Documents:**
Put your PDF files in the `data` folder of the repository,
for example `data/pdfs_files/` or any other location that is accessible (e.g. `home/john/data/pdfs_files/`).
Enter the above path in the text box and Click on "Submit PDFs" to ingest and process the documents.   
**demo files:** There are already few sample PDFs in the `data/sample_pdfs/` folder for a quick test of application.

2. **Ask a Question:**
If the PDF files have been ingested and the vector embeddings have been generated successfully,
the question box will appear.
Enter your question related to the uploaded documents.
Click on "Submit Question" to receive an answer based on the uploaded documents.

3. **View Q&A History:**
The application shows a history of your questions and answers for reference.
But it does not maintain that history for any follow up question.
   

### Parameter Descriptions
The application uses a YAML configuration file (config.yaml) to manage various settings. 
Below is an overview of the key configuration parameters:
- **embed_model_id**: Identifier for the embedding model used to generate vector representations of text chunks.
- **openai_modelID**: The OpenAI model ID used for generating responses (e.g., "gpt-4o").
- **top_k_BM25**: Number of top documents to retrieve using the BM25 retriever.
- **top_k_semantic**: Number of top documents to retrieve using semantic search.
- **top_k_final**: Number of top-ranked documents to use for generating the final answer.

---

## External Services

This application relies on the following externally hosted services:

1. **OpenAI API:**
   - **Usage:** For generating responses using GPT-4.
   - **Configuration:** Requires an `OPENAI_API_KEY` to authenticate requests.
   - **Notes:** Ensure you have an active OpenAI account and sufficient API credits. The API key should be set in the `.env` file as `OPENAI_API_KEY`.

2. **HuggingFace Models:**
   - **Usage:** For generating text embeddings and using cross-encoder models.
   - **Configuration:** No API key required for public models, but private models may require authentication.
   - **Notes:** Some models may have usage limits or require specific licensing. Ensure that the model names specified in `config.yaml` are accessible and properly configured.

3. **ChromaDB:**
   - **Usage:** Acts as the vector store for embedding data, enabling efficient similarity searches.
   - **Configuration:** Installed locally, but can be configured to use cloud-based instances if needed. The `persist_directory` and `collection_name` are specified in the `config.yaml` file.
   - **Notes:** Ensure that the `persist_directory` path is correctly set and that the application has read/write permissions to this directory.

4. **LangSmith (Optional):**
   - **Usage:** For tracing and monitoring the application's operations.
   - **Configuration:** Requires a `LANGSMITH_API_KEY` to authenticate requests.
   - **Notes:** This service is optional and can be skipped if not needed. If used, set the API key in the `.env` file as `LANGSMITH_API_KEY`.

5. **HuggingFace Embedding Models:**
   - **Usage:** Utilized for generating embeddings of text chunks to facilitate semantic search.
   - **Configuration:** Specified via `embed_model_id` in `config.yaml`.
   - **Notes:** Ensure that the chosen embedding model is suitable for your use case and that it is accessible without additional authentication if using public models.

6. **Cross-Encoder Models:**
   - **Usage:** Employed to score and rank both keyword-based and semantic search retrieval results.
   - **Configuration:** Model names are specified in `config.yaml` under `Retrieval.CE_model_name` and `Retrieval.semantic_CE_model`.
   - **Notes:** Select cross-encoder models that are compatible with your semantic search requirements. Verify model availability on HuggingFace and ensure they are properly integrated.

---

## Further Enhancements
This application is an initial demo version of such a system that can be considered as a starting point. 
It is not yet ready for production use and it also needs domain specific enhancements. 
The following points are important to consider for production use:

- **Scalable vectorstore:**
  The vectorstore can be stored in scalable cloud-based databases like Pinecone or other distributed solutions such as Milvus or Weaviate.
  This allows for more efficient retrieval of relevant text chunks when we deal with large volumes of data and advanced search algorithms.

- **More strict privacy settings:**
To ensure the uploaded documents are not exposed to the public, it is recommended to use a private cloud-based service like Google Cloud Storage or AWS S3. Also, the privacy of the llm model must be considered, specifically when used as an API service. Other data privacy measures must be taken into account regarding the vectore store and the embedding vectors.

- **Enhancing document loader for structured data:**
Depending on the type of reports you receive some information might be stored in tables, images or graphs alogn with the text in the pdf files.
Extracting such information needs a document loader that can handle structured data.

- **Using meta information for retrieval:**
It can always be useful to have meta information associated with documents, such as the title, author, date, or other information that can be useful in the context of the question. Such improvements can improve the retrival speed and relevance. 

- **Leveraging RAG history:**
In large scale production systems, it is efficient to store and use the previous QAs from previous users.
This can be enhance the RAG's performance in terms of concurrency and accuracy.

- **More domain specific adjustments:**
Some configurations depend on the type of pdf reports you usually revieve as well as the type of information you want to ask for.
This can influence the choice of chunk size, the chosen Large Language Model, and the number of retrieved documents in the pipeline.  
The AI developer needs to study the real PDF samples before deciding on these parameters.

- **Agentic pipeline:**
This project hesitated to use any Agentic pipeline to prevent over-engineering and to maintain the simplicity of the code.
However, in more complex use cases (e.g. large datasets, or high semantic ovelap between irrelevant documents) a more advanced pipeline such as Agentic design can add considerable improvement in terms of retrieval accuracy.