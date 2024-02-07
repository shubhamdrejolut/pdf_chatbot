import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
import os
import pinecone
import tempfile

load_dotenv()

# Set your Langchain API key here
llm = ChatOpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'), temperature=0.6)

# Function to extract tables from a PDF using camelot
def extract_tables_from_pdf(pdf_file):
    try:
        loader = PyPDFLoader(pdf_file)
        pages = loader.load_and_split()

        return pages
    except Exception as e:
        print(f"An error occurred while extracting tables: {e}")
        return None 
    
def create_embeddings():
            embeddings = OpenAIEmbeddings()
            return embeddings

def add_docs_to_pinecone(docs, namespace):
            pinecone.init(
                api_key=os.getenv("PINECONE_API_KEY"),
                environment=os.getenv("PINECONE_ENV"),
            )
            index_name = os.getenv("PINECONE_INDEX_NAME")

            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            document = text_splitter.split_documents(docs)

            embeddings = OpenAIEmbeddings()
            print('embeddings', embeddings)
            Pinecone.from_documents(document, embeddings, index_name=index_name, namespace=namespace)

            if index_name not in pinecone.list_indexes():
                print(pinecone.list_indexes())
                pinecone.create_index(name=index_name, metric="cosine", dimension=1536)

            return "true"

def query_existing_vectors(query, namespace):
            embeddings = create_embeddings()
            pinecone.init(
                api_key=os.getenv("PINECONE_API_KEY"),
                environment=os.getenv("PINECONE_ENV"),
            )
            index_name = os.getenv("PINECONE_INDEX_NAME")

            docsearch = Pinecone.from_existing_index(index_name, embeddings, namespace=namespace)
            retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":2})

            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
            result = qa({"query": query})
            return result['result']

def main():
    st.title("PDF Extraction and Vector Search")
    pdf_file_name = None

     # Allow user to upload a PDF file
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if pdf_file is not None:
        pdf_file_name = pdf_file.name.rsplit('.', 1)[0]  # Assuming pdf_file_name is sanitized and valid for use as a namespace
        print('pdf_file_name' ,pdf_file_name)

        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(pdf_file.getvalue())
            tmpfile_path = tmpfile.name

        # Now you have a file path you can use with extract_tables_from_pdf
        if st.button('Extract File'):
            docs = extract_tables_from_pdf(tmpfile_path)
            if docs is not None:
                add_docs_to_pinecone(docs, pdf_file_name)
                st.success("File extracted and processed successfully!")
            else:
                st.error("An error occurred during file processing.")


    # Streamlit interface for querying existing vectors in Pinecone
    query = st.text_input("Enter your query here")
    if query:
        query_result = query_existing_vectors(query, pdf_file_name)
        st.write(query_result)
        
if __name__ == "__main__":
    main()


