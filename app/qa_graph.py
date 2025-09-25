from typing import List, Tuple, TypedDict
from langgraph.graph import END, START, StateGraph
from llm_nodes import *
import time
import os
import re

# State definition
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    conversation_history: List[Tuple[str, str]]
    recursion_count: int
    last_step_time: float

# Helper function to format sources for display
def format_sources_for_display(documents) -> str:
    if not documents:
        return ""
    sources = {}
    for doc in documents:
        file_name = os.path.basename(doc.metadata.get('file_name', 'Unknown File'))[:-3]
        if file_name not in sources:
            sources[file_name] = {'file': file_name}
    source_lines = ["\n\nSources:"]
    for source in sources.values():
        source_lines.append(f"{source['file']}")
    return "\n".join(source_lines)

# Main graph builder function
def build_graph(retriever):

    # Document retrieval node
    def retrieve(state):
        step_start = time.time()
        print("\n--- Executing RETRIEVE Node ---")
        if "last_step_time" in state:
            duration = step_start - state["last_step_time"]
            print(f"Time since last step: {duration:.3f}s")

        docs = retriever.get_relevant_documents(state["question"])

        total_time = time.time() - step_start
        print(f"RETRIEVE completed in {total_time:.3f}s total\n")
        return {
            **state,
            "documents": docs,
            "last_step_time": time.time()
        }

    # Response generation
    def generate(state):
        step_start = time.time()
        if "last_step_time" in state:
            duration = step_start - state["last_step_time"]
            print(f"Time since last step: {duration:.3f}s")

        documents_text = ""
        for doc in state["documents"]:
            documents_text += doc.page_content

        llm_start = time.time()

        # Handle streaming vs non-streaming
        try:
            # Try streaming first
            chunks = []
            for chunk in generator.stream({
                "conversation_history": state["conversation_history"],
                "document": documents_text,
                "question": state["question"]
            }):
                chunks.append(chunk)
            llm_response = "".join(chunks)
        except:
            # Fallback to regular invoke
            llm_response = generator.invoke({
                "conversation_history": state["conversation_history"],
                "document": documents_text,
                "question": state["question"]
            })

        llm_time = time.time() - llm_start

        generation_text = llm_response if llm_response else "I could not generate a response."

        source_info = format_sources_for_display(state["documents"])
        generation_with_sources = generation_text + "\n\n" + source_info

        total_time = time.time() - step_start
        print(f"LLM generation took {llm_time:.3f}s")
        print(f"GENERATE completed in {total_time:.3f}s total\n")

        return {
            **state,
            "generation": generation_with_sources,
            "last_step_time": time.time()
        }

    # Query transformation node for improving retrieval
    def transform_query(state):
        step_start = time.time()
        if "last_step_time" in state:
            duration = step_start - state["last_step_time"]
            print(f"Time since last step: {duration:.3f}s")
        
        rewrite_start = time.time()
        better_q = question_rewriter.invoke({"question": state["question"]})
        rewrite_time = time.time() - rewrite_start
        
        total_time = time.time() - step_start
        print(f"Query rewrite took {rewrite_time:.3f}s")
        print(f"Original: '{state['question']}'")
        print(f"Rewritten: '{better_q}'")
        print(f"TRANSFORM_QUERY completed in {total_time:.3f}s total\n")
        
        return {**state, "question": better_q, "recursion_count": state["recursion_count"] + 1, "last_step_time": time.time()}

    # Response quality evaluation node
    def grade_generation(state):
        step_start = time.time()
        if "last_step_time" in state:
            duration = step_start - state["last_step_time"]
            print(f"Time since last step: {duration:.3f}s")
        
        print(f"Current recursion count: {state['recursion_count']}")
        
        if state["recursion_count"] >= 2:
            total_time = time.time() - step_start
            print(f"Recursion limit reached ({state['recursion_count']}), forcing end")
            print(f"GRADE_GENERATION completed in {total_time:.3f}s total\n")
            return "useful"
        
        documents_content_list = " ".join([doc.page_content for doc in state["documents"]]) if len(state["documents"]) > 0 else ""
        
        generation_to_check = state["generation"].split("\n\n**Source")[0]

        hallucination_start = time.time()
        h_score_response = hallucination_grader.invoke({
            "documents": documents_content_list,
            "generation": generation_to_check,
            "conversation_history": state["conversation_history"]
        })

        # Extract "yes" or "no" from the response
        h_score = "yes" if "yes" in h_score_response.lower() else "no"
        hallucination_time = time.time() - hallucination_start
        print(f"Hallucination check: {h_score} (took {hallucination_time:.3f}s)")
        
        if h_score == "no":
            total_time = time.time() - step_start
            print(f"Failed hallucination check")
            print(f"GRADE_GENERATION completed in {total_time:.3f}s total\n")
            return "not useful"
        
        answer_start = time.time()
        a_score_response = answer_grader.invoke({
            "question": state["question"],
            "generation": generation_to_check,
            "conversation_history": state["conversation_history"]
        })

        # Extract "yes" or "no" from the response
        a_score = "yes" if "yes" in a_score_response.lower() else "no"
        answer_time = time.time() - answer_start
        print(f"Answer quality check: {a_score} (took {answer_time:.3f}s)")
        
        decision = "useful" if a_score == "yes" else "not useful"
        total_time = time.time() - step_start
        print(f"Final decision: {decision}")
        print(f"GRADE_GENERATION completed in {total_time:.3f}s total\n")
        
        return decision
    
    # Workflow entry point for state initialization
    def entry_point(state):
        step_start = time.time()
        print("--- QA Graph Entry Point ---")
        
        history = state.get("conversation_history") 
        if not isinstance(history, list):
            history = []
        
        total_time = time.time() - step_start
        print(f"ENTRY_POINT completed in {total_time:.3f}s total\n")
    
        return {
            "question": state["question"],
            "documents": [],
            "conversation_history": history,
            "recursion_count": 0,
            "last_step_time": time.time()
        }
    
    # Graph construction and workflow definition
    wf = StateGraph(GraphState)
    wf.add_node("retrieve", retrieve)
    wf.add_node("generate", generate)
    wf.add_node("transform_query", transform_query)
    wf.add_node("unsure", lambda state: state)
    wf.add_node("entry_point", entry_point)

    # Workflow path definition
    wf.add_edge(START, "entry_point")
    wf.add_edge("entry_point", "retrieve")
    wf.add_edge("retrieve", "generate")
    wf.add_edge("transform_query", "retrieve")

    # Conditional routing
    wf.add_conditional_edges("generate", grade_generation, {
        "useful": END,
        "not useful": "transform_query"
    })

    wf.add_edge("unsure", END)
    
    return wf.compile()