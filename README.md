This is a Q&A chatbot that uses a method called RAG (Retrieval-Augmented Generation) to answer questions. It works by using LangChain to create RAG chains. It uses the Milvus database as a vector database to find relevant information for the query. Once it gets the relevant information from the vector database, it combines the question and the relevant information and sends them as input to ChatGPT-3.5 Turbo through an API.

The maximum length ChatGPT-3.5 Turbo can handle right now is 16,385 tokens, so you can share long documents with it. Its multilingual ability is especially valuable since this chatbot is mainly used in Turkish Law.

It uses query translation methods to improve accuracy, but sometimes they make the output too long, exceeding the 16,385 token limit. Sharing too much information with LLM can also lead to incorrect outputs. So, the most accurate way to get answers is by using a single query.

The program is still under development. Therefore, I will add more features in the future.

If you encounter any issues or have suggestions or want to contribute code, I would be very happy. Thank you for your interest in my project, and I appreciate your support!
