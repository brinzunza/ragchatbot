from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .llm_config import get_llm


# LLM instance with streaming support
# Note: When using OpenAI, native prompt caching is automatically used.
# When using Ollama (0.1.33+), KV cache reuse is automatically enabled
# when the prefix of the prompt remains identical between calls.
llm = get_llm(
    temperature=0,
    streaming=True,
    num_predict=2000,  # Only used by Ollama
    num_ctx=8192  # Only used by Ollama - ensure context window is large enough for caching
)


# QUESTION ANSWERING COMPONENTS

# Main response generator
# OPTIMIZATION: 'Knowledge Base' is placed BEFORE 'Conversation Context' 
# because it is the largest static block. If follow-up questions use the 
# same documents, the KV cache for the documents will be reused even as history grows.
generator_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert assistant trained to answer questions using the provided documents.
    You provide comprehensive technical guidance based on the uploaded documentation and context. Your answers
    should rely only on the content and context of this unless explicitly instructed otherwise. If the user's
    question falls outside the document's scope, politely inform them that the information is not covered. Never provide code or command line instructions.

    Your task is to answer the user's question based on the provided "Knowledge Base".

    Provide a detailed, comprehensive answer to the user's question. Be direct, comprehensive, thorough, and accurate.
    Provide a complete answer without preamble or closing remarks. Just provide the answer as plain text."""),
    ("user", """Knowledge Base: 
    {document}

    Conversation Context: 
    {conversation_history}

    Question: 
    {question}

    Answer:""")
])
generator = generator_prompt | llm | StrOutputParser()


# Hallucination detection component
hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", "Evaluate whether the Answer is mentioned in either the 'Documents' OR the 'Conversation Context'."),
    ("user", """Documents: 
    {documents}

    Conversation Context: 
    {conversation_history}

    Answer to evaluate: 
    {generation}

    Score 'no' if it contains claims not mentioned in or contradictions from either the provided Documents or Conversation Context.
    Else Score 'yes'.

    Return only {{"score": "yes"}} or {{"score": "no"}}""")
])
hallucination_grader = hallucination_prompt | llm | StrOutputParser()


# Answer quality evaluation component
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Does this answer adequately address the question? Consider the conversation context if relevant."),
    ("user", """Answer to evaluate: 
    {generation}

    Conversation context: 
    {conversation_history}

    Question: 
    {question}

    Score 'yes' if the answer:
    - Provides relevant detail
    - Is relevant to the conversation flow

    Score 'no' if off-topic.

    Return only {{"score": "yes"}} or {{"score": "no"}}""")
])
answer_grader = answer_prompt | llm | StrOutputParser()


# Query rewriting component
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rephrase the user's question so it is clear and self-contained for a search engine."),
    ("user", "Original Question: {question}\n\nOutput:")
])
question_rewriter = rewrite_prompt | llm | StrOutputParser()
