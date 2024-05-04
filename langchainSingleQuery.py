import config
from langchain import hub
from langchain_community.vectorstores import Milvus
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI


def format_docs(docs):
    return "\n".join(doc.page_content for doc in docs)


# Setup
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
prompt = hub.pull("ahmet-kaan-ozbek/icra_iflas_danismani")

# Collection names and QUERY which is the QUESTION
collection_names = ["icra_iflas_not_1", "icra_iflas_kanunu"]
query = ("Yukarıdaki olay çerçevesinde Sağlam Seramik Limited Şirketi’nin çeşitli şirketlerden alacakları olduğunu "
         "öğrenen Ali Veli’nin bu alacaklara haciz koyması mümkün müdür? Mümkünse bu haciz hangi şekillerde "
         "gerçekleşebilir ve hangi andan itibaren hüküm ifade eder?")
case = ("Av. Ali Veli ile yaptığı sözleşme uyarınca hukuki konularda danışmanlık hizmeti alan Sağlam Seramik Limited "
        "Şirketi, pandemi süreci nedeniyle maddi açıdan zorluğa düşmüş ve aylık hizmet bedeli olan 5.000 TL’yi "
        "ödeyemez duruma gelmiştir. Bunun üzerine Av. Ali Veli, Sağlam Seramik Limited Şirketi aleyhine icra takibi "
        "başlatmıştır. Sağlam Seramik Limited Şirketi’nin itiraz süresini kaçırması nedeniyle takip 19.12.2019 "
        "tarihinde kesinleşmiştir.")

all_formatted_docs = ""

# Processing and concatenating results of each collection
for collection_name in collection_names:
    vector_db = Milvus(
        embeddings,
        connection_args={"host": "localhost", "port": "19530"},
        collection_name=collection_name
    )
    retriever_results = vector_db.similarity_search(query + "\n" + case)
    formatted_docs = format_docs(retriever_results)
    all_formatted_docs += formatted_docs + "\n"

# Execute the RAG chain
rag_chain = (
        {"context": lambda x: all_formatted_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

print(rag_chain.invoke(case + "\n" + query))
