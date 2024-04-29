import config
from langchainSingleQuery import hub
from langchain_community.vectorstores import Milvus
from langchain_core.load import dumps, loads
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI


def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)


def get_unique_union(documents: list[list]):
    # Unique union of retrieved docs
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]


# Setup
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
prompt = hub.pull("ahmet-kaan-ozbek/icra_iflas_danismani")

# Multi Query: Different Perspectives
template = """You are an AI language model assistant. Your task is to generate three 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Give your answers in Turkish. 
Original question: {question}"""
prompt_perspectives = ChatPromptTemplate.from_template(template)

generate_queries = (
        prompt_perspectives
        | ChatOpenAI(temperature=0)
        | StrOutputParser()
        | (lambda x: x.split("\n"))
)

# Collection names and QUERY which is the QUESTION
collection_names = ["icra_iflas_not_1"]
query = "Borçluya verilen satış yetkisi hakkında detaylı bilgi paylaşabilir misin?"
all_documents = []

# Generate queries
query_variations = generate_queries.invoke({"question": query})

# Processing and concatenating results of each collection
for query_variation in query_variations:
    for collection_name in collection_names:
        vector_db = Milvus(
            embeddings,
            connection_args={"host": "localhost", "port": "19530"},
            collection_name=collection_name
        )
        retriever_results = vector_db.similarity_search(query_variation)
        all_documents.append(retriever_results)

# Get unique documents and format them
unique_documents = get_unique_union(all_documents)
all_formatted_docs = format_docs(unique_documents)

# Execute the RAG chain
rag_chain = (
        {"context": lambda x: all_formatted_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

print(rag_chain.invoke(query))
