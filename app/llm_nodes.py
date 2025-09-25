from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from typing import List, Tuple, Dict, Any
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
import json

# Model configuration
model = "llama3.2:latest"

# LLM instance
llm_str = ChatOllama(
    model=model,
    temperature=0,
    num_predict=2000,
    streaming=True
)

# QUESTION ANALYSIS COMPONENTS

# Main response generator (text-only)
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
generator = generator_prompt | llm_str | StrOutputParser()

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
hallucination_grader = hallucination_prompt | llm_str | StrOutputParser()

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
answer_grader = answer_prompt | llm_str | StrOutputParser()

# Query rewriting component
rewrite_prompt = PromptTemplate(
    template="""Rephrase this question below so it is clear and self-contained. 

    Original Question: {question}

    Output:""",
    input_variables=["question"],
)
question_rewriter = rewrite_prompt | llm_str | StrOutputParser()


# DATA ANALYSIS COMPONENTS

# Data file
data_file_path = "files/clean_data.csv"

# Planning prompt
planning_prompt_template = """You are a helpful AI assistant specialized in data analysis. You will receive a user question about a dataset and a list of the dataset's columns. Your task is to create a plan outlining the steps needed to answer the question using the available data. The plan must work in a sequential order so that it can be coded without issues.

    User Question: {question}

    Available Columns: {columns}

    DataFrame Columns and Types:
    {df_info}

    Return a JSON list of strings ONLY.  Do not include any other text or explanations.  Example: ["step 1", "step 2", "step 3"]
    JSON Output:"""
planning_prompt = PromptTemplate(template=planning_prompt_template, input_variables=["question", "columns", "df_info"])

# Function to create planning prompt
def create_planning_prompt(question: str, columns: List[str], df_info: str) -> str:
    prompt = planning_prompt.format(question=question, columns=columns, df_info=df_info)
    return prompt

# Function to create code generation prompt
def create_da_code_prompt(data_file_path: str, df_shape: tuple, df_columns: list, user_question: str, plan_steps: str, df_info: dict) -> str:
    # Get unique values for each column (up to 10)
    df = pd.read_csv(data_file_path)
    column_values = {}
    for column in df_columns:
        unique_values = df[column].dropna().unique()
        column_values[column] = unique_values[:10].tolist()
    
    # Convert column values to a readable string
    column_values_str = "\n".join([
        f"{col}: {vals}" for col, vals in column_values.items()
    ])

    df_info_serializable = {col: str(dtype) for col, dtype in df_info.items()}
    df_info_string = json.dumps(df_info_serializable)

    prompt = f"""
    You are a Python data analysis expert. Given a DataFrame 'df', a question, and a specific step from a data analysis plan, write Python code to execute that step.

    DataFrame Info:
    - Data: {data_file_path}
    - Shape: {df_shape}
    - Columns: {df_columns}
    
    Unique Column Values (up to 10 per column):
    {column_values_str}

    DataFrame Columns and Types:
    {df_info_string}

    User Question: {user_question}

    Plan Steps:
    {plan_steps}

    Requirements:
    1. Use 'df' for the DataFrame.
    2. Available libraries: pandas (pd), numpy (np), matplotlib.pyplot (plt), seaborn (sns).
    3. Only return executable Python code. No explanations or markdown.
    4. DO NOT USE code marks such as ```. 
    5. Print the final response. If it is a dataframe, print df.to_string().
    6. Do not use any libraries or attributes that are not explicitly allowed. For example, do not use attributes like .total_seconds() or .to_datetime()

    **NEVER CREATE YOUR OWN DATAFRAME. NO SYNTHETIC OR EXAMPLE DATA.**

    **ASSUME THAT THE DATA IS STORED IN A VARIABLE CALLED data**

    """
    return prompt

# Function to create analysis prompt
def create_da_analysis_prompt(user_question: str, result, plan: str, df: pd.DataFrame) -> str:
    print(f"result: {result}")

    output = result['output']

    context = f"Original Question: {user_question}\n"
    context += f"Plan: {plan}\n"
    
    if output:
        context += f"Text Output:\n{output}\n"
    else:
        context += "No output.\n"
    
    columns_string = ", ".join(df.columns.tolist())
    
    return f"""
    You are a data analysis expert. Keep the values raw and analyze the provided data objectively. The data comes from various sources and may include performance metrics, system traces, or other analytical information. Analyze the following results and provide a concise and comprehensive explanation on how this information is valuable. Focus on patterns, anomalies, and actionable insights. DO NOT SHOW CODE.

    These are the columns in the dataframe:
    {columns_string}

    Context: {context}

    If context is not provided, provide a polite error message.
    Do NOT add any additional text other than the analysis.
    Always start with the answer, then give analysis on whether it is good, bad, something, and explain why.
    Keep the analysis to a couple sentences.
    """