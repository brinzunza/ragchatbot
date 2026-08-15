from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm_config import get_llm


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
    ("system", """You are a helpful, knowledgeable assistant. Answer the user's question directly, drawing on
    your own general knowledge freely. You are also given a "Knowledge Base" of document excerpts the user
    has uploaded — when a fact, figure, or claim in your answer comes from the Knowledge Base, wrap exactly
    that span in markdown bold, like **this**. Do not bold anything that comes from your own general
    knowledge or reasoning — bold is reserved strictly for content sourced from the Knowledge Base. If the
    Knowledge Base is empty or irrelevant to the question, just answer normally with no bolding. Never provide
    code or command line instructions.

    Provide a detailed, comprehensive answer to the user's question. Be direct, comprehensive, thorough, and accurate.
    Provide a complete answer without preamble or closing remarks. Just provide the answer as plain text,
    using ** only to mark Knowledge-Base-sourced spans."""),
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
# The generator is now allowed to answer from general knowledge, so this only checks
# the spans the generator itself marked as Knowledge-Base-sourced (wrapped in **bold**).
# Unmarked, unbolded text is general knowledge and is not checked against the documents.
hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", """The Answer below may mix general knowledge with facts the model claims come from the
    Knowledge Base — those claims are wrapped in **bold**. Ignore all non-bold text entirely. For each
    **bolded** span only, check whether it is actually supported by the Documents or Conversation Context."""),
    ("user", """Documents:
    {documents}

    Conversation Context:
    {conversation_history}

    Answer to evaluate:
    {generation}

    Score 'no' only if a **bolded** span makes a claim not supported by or contradicting the Documents or
    Conversation Context. If there are no bolded spans, or all bolded spans are supported, score 'yes'.

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
