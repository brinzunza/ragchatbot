from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from llm_config import get_llm


def _chat_llm():
    """
    Build a fresh LLM instance reflecting whatever provider is currently
    configured. Called per chain build rather than once at import time, so
    switching between OpenAI and Ollama via the API key setting takes effect
    on the next request instead of requiring a backend restart.

    Note: When using OpenAI, native prompt caching is automatically used.
    When using Ollama (0.1.33+), KV cache reuse is automatically enabled
    when the prefix of the prompt remains identical between calls.
    """
    return get_llm(
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
    ("system", """You are a helpful, knowledgeable assistant. Answer the user's message directly, drawing on
    your own general knowledge freely. You are also given a "Knowledge Base" of document excerpts the user
    has uploaded — when a fact, figure, or claim in your answer comes from the Knowledge Base, wrap exactly
    that span in markdown bold, like **this**. Do not bold anything that comes from your own general
    knowledge or reasoning — bold is reserved strictly for content sourced from the Knowledge Base. Never
    provide code or command line instructions.

    If the Knowledge Base is empty, or does not clearly and directly relate to the message — including
    greetings, small talk, thanks, or questions on a different topic — treat it as if it were not there at
    all. Do not mention it, summarize it, or work it into your answer just because it was provided. Respond
    the way you normally would in conversation, with no bolding.

    Provide a detailed, comprehensive answer to the user's message. Be direct, comprehensive, thorough, and
    accurate. Provide a complete answer without preamble or closing remarks. Just provide the answer as
    plain text, using ** only to mark Knowledge-Base-sourced spans."""),
    ("user", """Knowledge Base:
    {document}

    Conversation Context:
    {conversation_history}

    Question:
    {question}

    Answer:""")
])


def get_generator():
    """Fresh generator chain, bound to whichever LLM is currently configured."""
    return generator_prompt | _chat_llm() | StrOutputParser()


# Relevance gate — runs on the retrieved excerpts before generation. Vector
# search always returns its top-k nearest neighbors even when none of them
# are actually relevant (e.g. a "hello" greeting still matches *something*
# in a homework PDF), and asking the generator to notice mid-answer that
# the Knowledge Base doesn't apply and quietly ignore it is unreliable,
# especially on smaller local models. A single up-front check is a much
# easier judgment to get right, and lets the caller drop the excerpts
# entirely (and skip showing sources) when nothing relevant is found.
#
# This went through two worse designs first, both disproven empirically
# against this app's own local model (llama3.2):
#   - A bare one-word yes/no verdict: with several concatenated excerpts
#     (each prefixed by an AI-generated summary, several describing an
#     unrelated section of the document), the model would say "no" even
#     when the real answer was present but buried a few lines down - a
#     needle-in-haystack miss with no room to actually look for it.
#   - Reasoning first, THEN a formatted "ANSWER: yes/no" verdict line: the
#     model reliably found and stated the correct fact on the reasoning
#     line, then routinely stopped before emitting the verdict line at
#     all, which a strict parser reads as a false "no".
# Asking it to do exactly one thing - state a supporting fact, or say
# "none" - plays to what it's actually reliable at (stating a fact when
# it finds one), and makes "found nothing" the single, simple negative
# case. It also deliberately biases toward relevance: a wrongly dropped
# real answer is worse than an occasional irrelevant excerpt reaching the
# generator, which already has its own instructions to ignore excerpts
# that don't actually apply.
relevance_prompt = ChatPromptTemplate.from_messages([
    ("system", """State one fact from the document excerpts below that helps answer the user's message,
    even if the connection is partial, indirect, or the fact is buried among unrelated text. If truly
    nothing in the excerpts relates to the message - for example if the message is a greeting, thanks, or
    small talk, or is about a completely different topic - respond with exactly: none"""),
    ("user", """Message:
    {question}

    Document excerpts:
    {document}""")
])


def get_relevance_checker():
    """Fresh relevance-gate chain, bound to whichever LLM is currently configured."""
    return relevance_prompt | _chat_llm() | StrOutputParser()


def parse_relevance_verdict(response: str) -> bool:
    """
    True unless the checker found nothing worth using. The prompt asks for
    exactly "none" in that case; treat only a response that's essentially
    just that word as a negative, so a trailing period or a slightly
    longer "none of these relate" doesn't get misread as it having found
    something.
    """
    cleaned = response.strip().strip(".!\"' ").lower()
    return not (cleaned == "none" or cleaned.startswith("none ") or cleaned.startswith("none,"))


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


def get_hallucination_grader():
    """Fresh hallucination-grading chain, bound to whichever LLM is currently configured."""
    return hallucination_prompt | _chat_llm() | StrOutputParser()


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


def get_answer_grader():
    """Fresh answer-quality-grading chain, bound to whichever LLM is currently configured."""
    return answer_prompt | _chat_llm() | StrOutputParser()


# Query rewriting component
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rephrase the user's question so it is clear and self-contained for a search engine."),
    ("user", "Original Question: {question}\n\nOutput:")
])


def get_question_rewriter():
    """Fresh query-rewriting chain, bound to whichever LLM is currently configured."""
    return rewrite_prompt | _chat_llm() | StrOutputParser()
