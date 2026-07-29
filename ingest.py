import json
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load the API key from your .env file into the environment
load_dotenv()
DOCS_DIR = "documents"
DB_DIR = "chroma_db"

# Written by chunk_experiment.py after it finds the best-performing combo.
BEST_PARAMS_FILE = "best_chunk_params.json"

# Fallback values, only used if chunk_experiment.py has never been run.
_DEFAULT_CHUNK_SIZE = 800
_DEFAULT_CHUNK_OVERLAP = 150
_DEFAULT_TOP_K = 4


def load_best_params():
    """Load chunk_size/chunk_overlap/top_k found by chunk_experiment.py.

    Falls back to hardcoded defaults (with a warning) if that script has
    never been run, so ingest.py still works out of the box.
    """
    if os.path.exists(BEST_PARAMS_FILE):
        with open(BEST_PARAMS_FILE) as f:
            params = json.load(f)
        print(
            f"Using experiment-derived params from {BEST_PARAMS_FILE}: "
            f"chunk_size={params['chunk_size']}, chunk_overlap={params['chunk_overlap']}, "
            f"top_k={params['top_k']} (hit_rate={params['hit_rate']})"
        )
        return params["chunk_size"], params["chunk_overlap"], params["top_k"]

    print(
        f"Warning: {BEST_PARAMS_FILE} not found — run chunk_experiment.py first "
        f"to determine real values. Falling back to unvalidated defaults: "
        f"chunk_size={_DEFAULT_CHUNK_SIZE}, chunk_overlap={_DEFAULT_CHUNK_OVERLAP}, "
        f"top_k={_DEFAULT_TOP_K}"
    )
    return _DEFAULT_CHUNK_SIZE, _DEFAULT_CHUNK_OVERLAP, _DEFAULT_TOP_K


CHUNK_SIZE, CHUNK_OVERLAP, TOP_K = load_best_params()
# TOP_K is not used during ingestion (ingest.py only builds the index, it
# doesn't query it) — it's loaded here so retrieval code can import it, e.g.
# Chroma(...).as_retriever(search_kwargs={"k": TOP_K})


def load_documents(folder):
    docs = []
    # Read every supported file in the folder into memory
    for name in os.listdir(folder):
      path = os.path.join(folder, name)
      if name.lower().endswith(".pdf"):
        docs.extend(PyPDFLoader(path).load())
      elif name.lower().endswith(".txt"):
        docs.extend(TextLoader(path, encoding="utf-8").load())
    
    print(f"Loaded {len(docs)} document pages.")
    return docs



def main():
    docs = load_documents(DOCS_DIR)

    # Split documents into overlapping chunks of about 800 characters.
    # Overlap keeps sentences from being cut awkwardly at boundaries.
    # RecursiveCharacterTextSplitter — splits long text into chunks 
    # by trying a list of separators
    #  (paragraphs, then lines, then sentences, then words, then characters) in order,
    #  so it splits at the most natural boundary it can.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    # Turn each chunk into an embedding (a meaning-vector)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Store the chunks + embeddings in Chroma, saved to disk
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )

    print(f"Done. Vector store saved to ./{DB_DIR}")


if __name__ == "__main__":
    main()