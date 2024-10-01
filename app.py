import streamlit as st
import git
from pathlib import Path
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

# Load Code Llama model
model_name = "codellama/CodeLlama-7b-Instruct-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

def chunk_code(content, chunk_size=1000):
    return [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]

def setup_repo(repo_url):
    repo_path = Path("./repo")
    if not repo_path.exists():
        git.Repo.clone_from(repo_url, repo_path)
    
    code_files = list(repo_path.rglob("*.[ch]")) + \
                 list(repo_path.rglob("*.java")) + \
                 list(repo_path.rglob("*.scala")) + \
                 list(repo_path.rglob("*.js")) + \
                 list(repo_path.rglob("*.cpp")) + \
                 list(repo_path.rglob("*.py"))
    code_contents = []
    for file in code_files:
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            chunks = chunk_code(content)
            code_contents.extend((str(file), chunk) for chunk in chunks)
    return code_contents

@st.cache_resource
def create_index(code_contents):
    embeddings = []
    for _, content in code_contents:
        inputs = tokenizer(content, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            embedding = model(**inputs).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        embeddings.append(embedding)
    
    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(np.array(embeddings))
    return index, code_contents

class EnhancedSearch:
    def __init__(self, index, code_contents):
        self.index = index
        self.code_contents = code_contents

    def search(self, query, k=10):
        query_embedding = model(**tokenizer(query, return_tensors="pt", truncation=True, max_length=512)).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        _, I = self.index.search(query_embedding.reshape(1, -1), k)
        return [self.code_contents[i] for i in I[0]]

def generate_response(query, relevant_code):
    prompt = f"Based on the following code:\n\n"
    for file_path, content in relevant_code:
        prompt += f"File: {file_path}\n{content}\n\n"
    prompt += f"Answer the following question: {query}"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(**inputs, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def paginate_code(code_chunks, page_size=5, page_number=1):
    start_index = (page_number - 1) * page_size
    end_index = start_index + page_size
    return code_chunks[start_index:end_index]

# Streamlit app
st.title("Code Q&A Chatbot for Large Repositories")

repo_url = st.text_input("Enter Git Repository URL")
if repo_url:
    try:
        with st.spinner("Setting up repository..."):
            progress_bar = st.progress(0)
            code_contents = setup_repo(repo_url)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            index, code_contents = create_index(code_contents)
        st.success("Repository indexed successfully!")
        
        search_engine = EnhancedSearch(index, code_contents)
        
        query = st.text_input("Ask a question about the code:")
        if query:
            results = search_engine.search(query)
            response = generate_response(query, results[:3])  # Use top 3 results for response generation
            st.write("Response:", response)
            
            st.write("Relevant code files:")
            page_number = st.number_input("Page", min_value=1, value=1)
            page_size = st.selectbox("Results per page", [5, 10, 20])
            paginated_results = paginate_code(results, page_size, page_number)
            
            for file_path, content in paginated_results:
                st.write(f"File: {file_path}")
                st.code(content, language="python")  # Adjust language based on file extension if needed
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")