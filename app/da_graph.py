import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import io
import traceback
import requests
import json
import os
import re
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END, START
from llm_nodes import create_da_code_prompt, create_da_analysis_prompt, create_planning_prompt
import time

# State definition for data analysis workflow
class DataAnalysisState(TypedDict):
    question: str
    generation: str
    plan: str
    code: str
    execution_result: Dict
    last_step_time: float

agent = None
data_file_path = "files/clean_data.csv"

# Data analysis state creation
class CustomDataAnalysisAgent:
    def __init__(self, file_path, llm_endpoint="http://localhost:11434/api/generate"):
        self.data_file_path = file_path
        self.df = pd.read_csv(file_path)
        self.llm_endpoint = llm_endpoint
        self.execution_context = {'df': self.df, 'pd': pd, 'plt': plt, 'sns': sns, 'np': np}

    
    def call_ollama(self, prompt, max_tokens=1000, model="llama3.2:latest"):
        payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.0, "max_tokens": max_tokens}}
        response = requests.post(self.llm_endpoint, json=payload, timeout=60)
        return response.json()['response']

    # Code execution function
    def execute_code(self, code):
        if not os.path.exists("files"):
            os.makedirs("files")
        
        old_stdout, sys.stdout = sys.stdout, io.StringIO()
        result = {'output_type': 'text', 'dataframe': None}
        
        try:
            exec(code, self.execution_context)
        except Exception as e:
            print(f"Error executing code: {e}")
        
        output = sys.stdout.getvalue().strip()
        sys.stdout = old_stdout
        
        print(f"Code Execution Output:\n{output}")
        
        result['output'] = output
        
        try:
            df = self.text_to_dataframe(output)
            if df is not None:
                result['output_type'] = 'dataframe'
                result['dataframe'] = df.to_json(orient='split')
        except Exception as e:
            print(f"Failed to parse DataFrame from output: {e}")
        
        return result

    def text_to_dataframe(self, text):
        from io import StringIO
        
        df = pd.read_csv(StringIO(text), sep=r'\s+', engine='python')
        return df

# Graph node functions, each representing a step in the workflow

def plan(state: DataAnalysisState):
    step_start = time.time()
    print("\n--- Executing PLAN Node ---")
    if "last_step_time" in state:
        duration = step_start - state["last_step_time"]
        print(f"Time since last step: {duration:.3f}s")

    planning_prompt_string = create_planning_prompt(state["question"], agent.df.columns.tolist(), agent.df.dtypes.to_dict())
    plan_response = agent.call_ollama(planning_prompt_string, model="llama3.2:latest")

    total_time = time.time() - step_start
    print(f"PLAN completed in {total_time:.3f}s total\n")
    return {**state, "plan": plan_response, "last_step_time": time.time()}


def generate_code(state: DataAnalysisState):
    step_start = time.time()
    print("\n--- Executing GENERATE_CODE Node ---")
    if "last_step_time" in state:
        duration = step_start - state["last_step_time"]
        print(f"Time since last step: {duration:.3f}s")

    code_prompt = create_da_code_prompt(
        data_file_path=agent.data_file_path,
        df_shape=agent.df.shape,
        df_columns=agent.df.columns.tolist(),
        user_question=state["question"],
        plan_steps=state["plan"],
        df_info=agent.df.dtypes.to_dict()
    )
    code_from_llm = agent.call_ollama(code_prompt, model="gpt-oss")
    code = code_from_llm.strip()
    code = re.sub(r'python', '', code)
    code = re.sub(r'```', '', code)

    print(f"Generated Code: \n{code}\n")
    code = "import pandas as pd\ndata = pd.read_csv('files/clean_data.csv')\n" + code

    total_time = time.time() - step_start
    print(f"GENERATE_CODE completed in {total_time:.3f}s total\n")
    return {**state, "code": code, "last_step_time": time.time()}


def execute_code(state: DataAnalysisState):
    step_start = time.time()
    print("\n--- Executing EXECUTE_CODE Node ---")
    if "last_step_time" in state:
        duration = step_start - state["last_step_time"]
        print(f"Time since last step: {duration:.3f}s")

    result = agent.execute_code(state["code"])

    print(f"Code Execution Output:\n\n{result['output']}\n```")

    total_time = time.time() - step_start
    print(f"EXECUTE_CODE completed in {total_time:.3f}s total\n")
    return {**state, "execution_result": result, "last_step_time": time.time()}


def analyze_results(state: DataAnalysisState):
    step_start = time.time()
    print("\n--- Executing ANALYZE_RESULTS Node ---")
    if "last_step_time" in state:
        duration = step_start - state["last_step_time"]
        print(f"Time since last step: {duration:.3f}s")

    analysis_prompt = create_da_analysis_prompt(state["question"], state["execution_result"], state["plan"], agent.df)
    comprehensive_analysis = agent.call_ollama(analysis_prompt, max_tokens=1500)

    total_time = time.time() - step_start
    print(f"ANALYZE_RESULTS completed in {total_time:.3f}s total\n")
    return {**state, "generation": comprehensive_analysis, "last_step_time": time.time()}


def build_da_graph(file_path: str):
    global agent
    agent = CustomDataAnalysisAgent(file_path=file_path)

    workflow = StateGraph(DataAnalysisState)

    workflow.add_node("plan", plan)
    workflow.add_node("generate_code", generate_code)
    workflow.add_node("execute_code", execute_code)
    workflow.add_node("analyze_results", analyze_results)

    workflow.add_edge(START, "plan")
    workflow.add_edge("plan", "generate_code")
    workflow.add_edge("generate_code", "execute_code")
    workflow.add_edge("execute_code", "analyze_results")
    workflow.add_edge("analyze_results", END)

    return workflow.compile()