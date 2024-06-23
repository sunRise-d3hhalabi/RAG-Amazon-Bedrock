# PDF RAG Reader utilizing AWS Bedrock

The PDF RAG Reader is a comprehensive project that harnesses the power of the Retrieval-Augmented Generation (RAG) model on enterprise data using the Faiss vector database, LangChain, and large language models provided by AWS Bedrock. The application was deployed in AWS EC2 Ubuntu virtual machine. The Faiss vector database stores and retrieves vectors associated with PDF documents efficiently. This approach enables the extraction of essential information from PDF files without training the model on question-answering datasets. The project has a user-friendly interface (utilizing Streamlit) for interacting with the PDF reader, making it accessible to users with various technical expertise.

Tools: Python 3.8, Langchain, Streamlit, AWS Bedrock, AWS EC2, Facebook AI Similarity Search (Faiss)

## How to run?

###  1. Create a new environment

```bash
conda create -n llmapp python=3.8 -y 
```

###  2. Activate the environment
```bash
conda activate llmapp 
```

###  3. Install the requirements package
```bash
pip install -r requirements.txt
```

###  4. run your application
```bash
streamlit run main.py
```