import config
from langchain import hub
from langchain_community.vectorstores import Milvus
from langchain_core.load import loads, dumps
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

# Setup for embeddings and OpenAI model
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
prompt = hub.pull("ahmet-kaan-ozbek/icra_iflas_danismani")

# Decomposition Template
template = """You are a helpful assistant that generates multiple sub-questions related to an input question. 
Generate multiple search queries in Turkish related to: {question} 
Output (2 queries):"""
prompt_decomposition = ChatPromptTemplate.from_template(template)

# Decomposition Chain
generate_queries_decomposition = (
        prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n"))
)

# Main question for decomposition
main_question = "Borçluya verilen satış yetkisi hakkında detaylı bilgi paylaşabilir misin?"

# Generate sub-questions
sub_questions = generate_queries_decomposition.invoke({"question": main_question})


# Function to format documents for concatenation
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


# Collect documents from different collections
all_documents = []
for sub_question in sub_questions:
    sub_question_docs = []
    collection_names = ["icra_iflas_not_1", "icra_iflas_kanunu"]
    for collection_name in collection_names:
        vector_db = Milvus(
            embeddings,
            connection_args={"host": "localhost", "port": "19530"},
            collection_name=collection_name
        )
        retriever_results = vector_db.similarity_search(sub_question)
        sub_question_docs.extend(retriever_results)
    all_documents.append(sub_question_docs)

# Apply the get_unique_union function
unique_documents = get_unique_union(all_documents)

# Format unique documents for concatenation
all_formatted_docs = format_docs(unique_documents)

# RAG Chain to generate final answer using all gathered context
rag_chain = (
        {"context": lambda x: all_formatted_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

# Print the final answer
final_answer = rag_chain.invoke(main_question)
print(final_answer)
