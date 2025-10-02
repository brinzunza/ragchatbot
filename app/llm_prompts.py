from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser


# Model configuration
MODEL_NAME = "llama3.2:latest"

# LLM instance with streaming support
llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0,
    num_predict=2000,
    streaming=True
)


# QUESTION ANSWERING COMPONENTS

# Main response generator
generator_prompt = PromptTemplate(
    template="""You are an expert assistant trained to answer questions using the provided documents.
    You provide comprehensive technical guidance based on the uploaded documentation and context. Your answers
    should rely only on the content and context of this unless explicitly instructed otherwise. If the user's
    question falls outside the document's scope, politely inform them that the information is not covered. Never provide code or command line instructions.

    Your task is to answer the user's question based on the provided "Knowledge Base".

    Provide a detailed, comprehensive answer to the user's question. Be direct, comprehensive, thorough, and accurate.
    Provide a complete answer without preamble or closing remarks. Just provide the answer as plain text.

    Conversation Context: {conversation_history}

    Knowledge Base: {document}

    Question: {question}

    Answer:""",
    input_variables=["conversation_history", "document", "question"]
)
generator = generator_prompt | llm | StrOutputParser()


# Hallucination detection component
hallucination_prompt = PromptTemplate(
    template="""Evaluate whether the Answer is mentioned in either the "Documents" OR the "Conversation Context". The answer can be found separated between the Documents or Conversation Context, they do not need to be found together.

    Documents: {documents}

    Conversation Context: {conversation_history}

    Answer to evaluate: {generation}

    Score 'no' if it contains claims not mentioned in or contradictions from either the provided Documents or Conversation Context.
    Else Score 'yes'.

    {{"score": "yes"}} or {{"score": "no"}}""",
    input_variables=["generation", "documents", "conversation_history"],
)
hallucination_grader = hallucination_prompt | llm | StrOutputParser()


# Answer quality evaluation component
answer_prompt = PromptTemplate(
    template="""Does this answer adequately address the question? If the conversation context is relevant, consider it.

    Answer to evaluate: {generation}

    Conversation context: {conversation_history}

    Question: {question}

    Score 'yes' if the answer:
    - Provides relevant detail
    - Is relevant to the conversation flow

    Score 'no' if off-topic.

    {{"score": "yes"}} or {{"score": "no"}}""",
    input_variables=["generation", "question", "conversation_history"],
)
answer_grader = answer_prompt | llm | StrOutputParser()


# Query rewriting component
rewrite_prompt = PromptTemplate(
    template="""Rephrase this question below so it is clear and self-contained.

    Original Question: {question}

    Output:""",
    input_variables=["question"],
)
question_rewriter = rewrite_prompt | llm | StrOutputParser()
