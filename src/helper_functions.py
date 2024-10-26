import os
import streamlit as st
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever                
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_loaders import PyPDFLoader
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers import SelfQueryRetriever
from sentence_transformers.cross_encoder import CrossEncoder
import chromadb


# ------------------------------
# Define the PDFManager class
# ------------------------------
class PDFManager:
    """
    Manages PDF loading, chunking, and vector store creation.
    """
    def __init__(self, pdf_path: str, config):
        """
        Initializes the PDFManager with the necessary configurations.

        Args:
            pdf_path (str): Path to the directory containing PDF files.
            config (OmegaConf): Configuration object containing model and vector store settings.

        Attributes:
            pdf_path (str): Stores the path to the PDF directory.
            embed_model_id (str): ID of the embedding model to be used.
            persist_directory (str): Directory for persisting the vector store.
            collection_name (str): Name of the collection in the vector store.
            documents (list): List to store loaded documents.
            vectorstore (Optional): Vector store object, initialized as None.
            small_chunks (Optional): Placeholder for small document chunks, initialized as None.
            large_chunks (Optional): Placeholder for large document chunks, initialized as None.
        """
        self.pdf_path = pdf_path
        # self.config = config
        self.embed_model_id = config.llm.embed_model_id        
        self.persist_directory = config.Vectorstore.persist_directory
        self.collection_name = config.Vectorstore.collection_name
        self.documents = []
        self.vectorstore = None        
        self.small_chunks = None
        self.large_chunks = None        

    def load_pdfs(self):            
        """
        Loads all PDF files from the specified directory using LangChain's PyPDFLoader.        

        Attributes:
            documents (list): List of loaded documents, updated in-place.
        """
        # filenames = os.listdir(self.pdf_path)
        # print(filenames)
        # metadata = [dict(source=filename) for filename in filenames]

        try:
            filenames = [file for file in os.listdir(self.pdf_path) if file.lower().endswith('.pdf')]
            if not filenames:
                st.warning("No PDF files found in the specified directory.")
                return

            docs = []
            for idx, file in enumerate(filenames):
                loader = PyPDFLoader(f'{self.pdf_path}/{file}')
                document = loader.load()
                for page_num, document_fragment in enumerate(document, start=1):
                    document_fragment.metadata = {"name": file, "page": page_num}
                    
                # print(f'{len(document)} {document}\n')
                docs.extend(document)
            self.documents = docs            
            st.info(f"Total document pages loaded: {len(self.documents)} from {self.pdf_path}")
        except Exception as e:
            st.error(f"Failed to load PDF files: {e}")
            return

    def chunk_documents(self):        
        """
        Splits loaded documents into small and large chunks using LangChain's RecursiveCharacterTextSplitter.

        Splits are performed with two different configurations: smaller chunks with no overlap and larger chunks with some overlap.

        Attributes:
            small_chunks (list): Stores the smaller document chunks.
            large_chunks (list): Stores the larger document chunks.

        Raises:
            Exception: If the document splitting process fails, an error message is displayed.
        """
        if not self.documents:
            st.error("No documents to split. Please load PDFs first.")
            return
        
        try:
            child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
            self.small_chunks = child_text_splitter.split_documents(self.documents)
            # print(len(self.small_chunks), len(self.small_chunks[0].page_content))

            large_text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            self.large_chunks = large_text_splitter.split_documents(self.documents)
            # print(len(self.large_chunks), len(self.large_chunks[0].page_content))

            st.info(f"Documents split into {len(self.small_chunks)} small and {len(self.large_chunks)} large chunks.")
        except Exception as e:
            st.error(f"Failed to split documents: {e}")
    
    def create_vectorstore(self):
        """
        Creates a vector store from the loaded document chunks using Chroma and HuggingFace embeddings.

        This function initializes an embedding model and a persistent Chroma client. It attempts to delete any existing 
        collection with the specified collection name before creating a new vector store. The vector store is created 
        using the large document chunks and is stored persistently.

        Attributes:
            vectorstore (Chroma): The created vector store containing the document embeddings.

        Raises:
            Exception: If there is an error during the creation of the vector store, an error message is displayed.
        """
        if not self.documents:
            st.error("No documents to index. Please load PDFs first.")
            return

        try:            
            embedding = HuggingFaceEmbeddings(model_name=self.embed_model_id)                                    
            # print(len(self.large_chunks))
            chroma_client = chromadb.PersistentClient(path=self.persist_directory)
            try:
                chroma_client.delete_collection(self.collection_name)
                print(f'Collection {self.collection_name} is deleted')
            except Exception:
                print(f'Collection {self.collection_name} does not exist')
            # print(len(chunks))
            self.vectorstore = Chroma.from_documents(
                documents=self.large_chunks,
                embedding=embedding,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name
            )

            collection = chroma_client.get_collection(name=self.collection_name)
            # st.info(f'Collection {collection_name} is created, number of itmes: {collection.count()}')

            st.success(f"Vectorstore {self.collection_name} created successfully with {collection.count()} documents.")
        except Exception as e:
            st.error(f"Failed to create vectorstore: {e}")
  
# ------------------------------
# Define the QA Retriever Function
# ------------------------------
class Retrievers:
    """
    Sets up various retrievers for keyword and semantic search.
    """
    def __init__(self, pdf_manager: PDFManager, config):        
        """
        Initialize the retriever with the vectorstore and small chunks of documents

        Args:
            pdf_manager (PDFManager): The PDFManager instance
            config (Config): The configuration object
        """

        self.vectorstore = pdf_manager.vectorstore
        self.small_chunks = pdf_manager.small_chunks
        self.modelID = config.llm.openai_modelID
        self.top_k_BM25 = config.Retrieval.top_k_BM25   
        self.top_k_semantic = config.Retrieval.top_k_semantic 
        self.CE_model_name = config.Retrieval.CE_model_name        
        self.llm = None        
        self.keyword_retriever = None
        self.retriever_large = None

    def setup_retrievers(self):          
        """
        Sets up the retrievers.

        Sets up the BM25 retriever and the large retriever based on the vectorstore and small chunks of documents.
        """
        if not self.small_chunks:
            st.error("No small_chunks to index. Please load PDFs first.")
            return
        try:
            retriever_bm25 = BM25Retriever.from_documents(documents=self.small_chunks)
            cross_encoder_model = HuggingFaceCrossEncoder(model_name=self.CE_model_name)
            reranker = CrossEncoderReranker(model=cross_encoder_model, top_n=self.top_k_BM25)
            self.keyword_retriever = ContextualCompressionRetriever(
                base_compressor=reranker,
                base_retriever=retriever_bm25
            )
                
            self.llm = ChatOpenAI(temperature = 0.0, model=self.modelID)

            metadata_field_info = [
                AttributeInfo(
                    name="name",
                    description="The name of the document",
                    type="string",
                )
            ]
            document_content_description = "financial consulting information"
            
            self.retriever_large = SelfQueryRetriever.from_llm(
                self.llm, 
                self.vectorstore, 
                document_content_description, 
                metadata_field_info, 
                verbose=True,
                search_kwargs={"k": self.top_k_semantic},
            )

            st.success("Retrievers created successfully.")
        except Exception as e:
            st.error(f"Failed to create retrievers: {e}")

class QAchains:
    """
    Handles the Question-Answering pipeline, including question shortening, retrieval, ranking, and answer generation.
    """
    def __init__(self, retrievers: Retrievers, config):        
        """
        Initializes the QAchain object.

        Args:
            retrievers (Retrievers): The object containing the retrievers.
            config (Config): The configuration object containing the necessary settings.
        """
        
        self.semantic_CE_model = config.Retrieval.semantic_CE_model
        self.top_k_final = config.Retrieval.top_k_final
        self.llm = retrievers.llm
        self.keyword_retriever = retrievers.keyword_retriever
        self.retriever_large = retrievers.retriever_large        
        self.question = None
        self.shortened_question = None
        self.retrieved_docs = None
        self.top_score_docs = None

    def shorten_question(self, question: str):
        """
        Shortens the question to a short phrase with essential keywords.

        Uses a ChatLLM to generate a shortened version of the question. The prompt is a description of the task of
        shortening the question with essential keywords. The shortened question is then used to retrieve relevant
        documents.

        Args:
            question (str): The original question to be shortened.

        Raises:
            Exception: If there is an error during the generation of the shortened question, an error message is displayed.
        """
        self.question = question

        shortening_prompt = """
        You are an expert financial advisor tasked with shortening the original question. 
        Your role is to reformulate the original question to short phrases with essential keywords.
        Mostly focus on company names, consultant or advisor names.
        The answer does not need to be complete sentense.
        Do not convert words to abbreviations.

        Original Question: "{original_question}"

        Reformulated phrases: """
        
        try:                        
            prompt_template = PromptTemplate(
                input_variables=["original_question"],
                template=shortening_prompt
            )
            
            shortening_chain = LLMChain(
                llm=self.llm,
                prompt=prompt_template
            )

            self.shortened_question = shortening_chain.run({"original_question": self.question})
            print(self.shortened_question)
            st.info("The question was shortened")

        except Exception as e:
            st.error(f"Failed to generate shortened question: {e}")
    
    def retrieve_chunks(self):          
        """
        Retrieves the relevant chunks of documents based on the shortened question.

        First, the method uses the keyword retriever to retrieve the relevant documents based on the shortened question.
        Then, it forms a question property by joining the document names and the original question.
        Finally, it uses the large retriever to retrieve the relevant chunks of documents based on the question property.

        The retrieved chunks are stored in the `retrieved_docs` attribute.
        """
        try:
            self.key_chunks = self.keyword_retriever.invoke(self.shortened_question)
            for doc in self.key_chunks:
                print(doc.metadata['name'], doc.metadata['page'])        
            key_documents = set()
            [key_documents.add(doc.metadata['name']) for doc in self.key_chunks]
            print(key_documents)
            key_documents = list(key_documents)

            question_prop = f'''
            According to the document name {" and the document name ".join(file for file in key_documents)}:
            {self.question}
            '''

            print(question_prop)
            self.retrieved_docs = self.retriever_large.invoke(question_prop)

        except Exception as e:
            st.error(f"Failed to retrieve documents: {e}")

    def rank_chunks(self):
        """
        Rank the retrieved chunks based on their relevance to the question.

        Uses the cross-encoder model to rank the retrieved chunks based on their relevance to the question.
        The top-k ranked documents are stored in the `top_score_docs` attribute.

        :return: None
        """
        try:
            model = CrossEncoder(self.semantic_CE_model)        

            all_retrieved_docs = self.retrieved_docs+self.key_chunks
            passages = [doc.page_content for doc in all_retrieved_docs]
            ranks = model.rank(self.question, passages)                
            for rank in ranks:
                print(f"{rank['score']:.2f}\t{all_retrieved_docs[rank['corpus_id']].metadata}")
            self.top_score_docs = [all_retrieved_docs[rank['corpus_id']] for rank in ranks[:self.top_k_final]]
            st.success(f"Top {self.top_k_final} ranked documents retrieved successfully.")

        except Exception as e:
            st.error(f"Failed to rank documents chunks: {e}")

    def generate_answer(self):
        """
        Generate an answer to the question based on the top-k ranked chunks of documents.

        Uses the LangChain's RetrievalQA to generate an answer based on the top-k ranked chunks of documents.
        The answer is generated using a custom prompt template that provides context from the top-k ranked documents.
        The answer is then parsed and returned as a string.

        :return: str
        """

        template = """ You are an expert financial analyst with extensive experience in interpreting reports, analyzing financial data, and generating insights from dense textual information. 
        Your task is to answer questions using only the provided document chunks as context. 
        Your answers should focus solely on the information within the document chunks and avoid speculation or any information not directly supported by the text.
        The document context provided includes various financial reports, business analyses, and forecasting values. 
        Your role is to deliver concise, well-supported responses that draw from this context, aligning with the standards and depth of a financial consultant.
        Always mention any value or number you find in the context that is relevant to the question.
        Also mention any non-numeric information that can clarify the financial context related to the question.
        If you don't know the answer based on the provided context, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}

        Expert financial answer:
        """

        try:
            custom_rag_prompt = PromptTemplate.from_template(template)
            chain = custom_rag_prompt | self.llm | StrOutputParser()
            
            context = "\ndocument_separator: <<<<>>>>>\n".join(doc.page_content for doc in self.top_score_docs)
            response = chain.invoke({"context": context, "question": self.question})
            return response.strip()
        except Exception as e:
            st.error(f"Failed to generate answer: {e}")