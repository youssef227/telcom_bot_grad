from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
import torch
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore import InMemoryDocstore
from transformers import RagTokenizer, RagRetriever, AutoModelForCausalLM, AutoTokenizer
import faiss
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Suppress the warning if necessary
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)

# Load and preprocess the PDF document
book_path = "VOVOVO_2024 - Google Docs.pdf"
loader = PyPDFLoader(book_path)
pages = loader.load_and_split()

# Clean pages
for page in pages:
    content = page.page_content
    content = content.replace("\n", "").replace("\t", "").replace("-", "")
    page.page_content = content

# Split text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200, length_function=len)
splitted_docs = splitter.transform_documents(pages)

# Collect texts
texts = [doc.page_content for doc in splitted_docs]

# Create embeddings using SentenceTransformer
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Initialize FAISS index
index = faiss.IndexFlatL2(384)

# Initialize the document store
docstore = InMemoryDocstore()

# Create FAISS vector store
vector_store = FAISS(embedding_function=embeddings.embed_query, index=index, docstore=docstore, index_to_docstore_id={})

# Index documents with the FAISS retriever
vector_store.add_texts(texts)

# Initialize the retriever
retriever = vector_store.as_retriever()


# Load PEFT configuration
config = PeftConfig.from_pretrained("youssef227/llama-3-8b-Instruct-bnb-telcom-3")
print("Step 1: Loaded PEFT configuration")

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained("unsloth/llama-3-8b-Instruct-bnb-4bit")
print("Step 2: Loaded base model")

# Apply PEFT configuration to the base model
model = PeftModel.from_pretrained(base_model, "youssef227/llama-3-8b-Instruct-bnb-telcom-3")
print("Step 3: Applied PEFT configuration to the base model")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-Instruct-bnb-4bit")
print("Step 4: Loaded tokenizer")

# Define the prompt template
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

def generator(text):
    # Define the context if it's used in the prompt

    # Use the retriever to get relevant documents
    retrieval_results = retriever.get_relevant_documents(text)
    retrieved_texts = [result.page_content for result in retrieval_results]

    # Concatenate retrieved texts into a single string
    context = " ".join(retrieved_texts)
    # Prepare the inputs
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                f" {context}انت ممثل خدمة العملاء لدى شركة فودافون.و دي معلومات ممكن تفيدك", # instruction
                text, # input
                "", # output - leave this blank for generation!
            )
        ], return_tensors="pt"
    ).to("cuda")

    # Generate the output
    outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)

    # Decode the output
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data['text']
    generated_text = generator(text)
    return jsonify({"generated_text": generated_text})

if __name__ == '__main__':
    app.run(debug=True)
 