# Import required libraries
import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

import os
import openai

# Load the OpenAI API key from the environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
load_dotenv()

# Load documents from a CSV file for the chatbot's knowledge base
loader = CSVLoader(file_path="ME2.csv")
documents = loader.load()

# Initialize embeddings and create a FAISS index for similarity search
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# Function to retrieve information based on a query
def retrieve_info(query):
    # Perform a similarity search and return the top 3 similar documents
    similar_response = db.similarity_search(query, k=3)
    page_contents_array = [doc.page_content for doc in similar_response]
    return page_contents_array

# Initialize the ChatOpenAI model with specific settings
llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")

# Define a template for the chatbot's responses
template = """Add 2+2"""

# Create a PromptTemplate object with the defined template
prompt = PromptTemplate(
    input_variables=["question", "relevant_data"],
    template=template
)

# Create a chain of language model and prompt for generating responses
chain = LLMChain(llm=llm, prompt=prompt)

# Function to generate response based on the user's question
def generate_response(question):
    relevant_data = retrieve_info(question)
    response = chain.run(question=question, relevant_data=relevant_data)
    return response

# Main function to run the Streamlit app
def main():
    # Set up the Streamlit page configuration and customize it
    st.set_page_config(
        page_title="Get to know me", page_icon=":male-technologist:")

    # Streamlit UI layout with columns
    col1, col2, col3 = st.columns([1, 2, 1])
    col1.header("Get to know me")
    col2.image("memoji.png", width=200)
    with open("resume.pdf", "rb") as file:
        col3.download_button(label="Download my Resume", data=file, file_name="resume.pdf", mime="application/pdf")

    # Input area for user to ask questions
    message = st.text_area("Hi, I am ABC. What would you like to know about me.")

    # Process and display the response
    if message:
        st.write("Typing...")
        result = generate_response(message)
        st.info(result)

# Entry point for the Streamlit application
if __name__ == '__main__':
    main()