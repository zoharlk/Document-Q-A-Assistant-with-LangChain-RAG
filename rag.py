# This is the heart of the project. It loads the index, retrieves the most relevant chunks
# for a question, builds a careful prompt, asks the model,
#  and returns the answer plus the sources it used.


from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model

from ingest import DB_DIR, TOP_K

load_dotenv()

# The instructions we send the model along with the retrieved text.
# Telling it to use ONLY the context is what reduces made-up answers.
PROMPT = """You are a helpful assistant. Answer the question using ONLY the
context below. If the answer is not in the context, say:
"I don't know based on the provided documents."
Cite the source filenames you used.
Context:

{context}

Question: {question}

Answer:"""


def get_retriever(k=TOP_K):
    # Re-open the saved Chroma store and turn it into a retriever
    # that returns the top k most relevant chunks.
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    return store.as_retriever(search_kwargs={"k": k})


def format_docs(docs):
    # Combine retrieved chunks into one labelled context string
    blocks = []
    for d in docs:
        # pulls the source filename that was attached to that chunk back at ingest time,
        #  defaulting to "unknown" if missing.
        source = d.metadata.get("source", "unknown")
        blocks.append(f"[Source: {source}]\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


def answer_question(question, k=TOP_K):
    retriever = get_retriever(k)
    
    # embeds the question, runs similarity search against the stored chunk vectors,
    # returns the top k Document objects. This is the "R" (retrieval) in RAG.
    docs = retriever.invoke(question)  # find relevant chunks
    
    context = format_docs(docs)
    
    # loads the chat model. temperature=0 means deterministic,
    # least-creative output — appropriate for a fact-answering
    # task where you want consistency over variety.
    model = init_chat_model("openai:gpt-4o-mini", temperature=0)
    
    # fills the two placeholders in the template,
    # producing the final instruction text sent to the model.
    prompt = PROMPT.format(context=context, question=question)
    
    # the "G" (generation) in RAG: sends the prompt to the LLM
    # and gets back a response object.
    response = model.invoke(prompt)  # ask the model
    
    # returns two things: the plain-text answer (response.content),
    # and the original list of retrieved Document objects (docs)
    return response.content, docs 


if __name__ == "__main__":
    q = input("Ask a question about your documents: ")
    answer, sources = answer_question(q)
    print("\n=== Answer ===\n", answer)
    print("\n=== Sources used ===")
    for d in sources:
        print("-", d.metadata.get("source", "unknown"))

