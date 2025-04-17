"""
BIAgent_Node.py

This module defines the BI Agent that is responsible for performing exploratory data analysis,
generating Python code to analyze shipment data, creating visualizations, and providing the final answer.
The agent loads its prompt from the prompt_templates folder.
"""
import openai
import streamlit as st
from src.core.bi_functions.bi_function import execute_codes
from src.core.order_consolidation.dynamic_consolidation import get_parameters_values
from src.utils.openai_api import get_supervisor_llm
from src.core.order_consolidation.consolidation_ui import timed_call


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage
from langchain.tools import tool


import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


@tool
def execute_analysis(df, response_text):
    """
    Executes analysis on the given dataframe using the response text.
    """
    results = execute_codes(df, response_text)
    return results


llm = get_supervisor_llm()

class BIAgent_Class:
    def __init__(self, llm, prompt, tools, data_description, dataset, helper_functions=None):
        """
        Initialize an Agent with the required properties.

        Parameters:
        - prompt (str): The prompt that defines the agent's task or behavior.
        - tools (list): The tools that the agent has access to (e.g., APIs, functions, etc.)
        - data_description (str): A description of the dataset the agent will work with.
        - dataset (dict or pd.DataFrame): The dataset that the agent will use.
        - helper_functions (dict, optional): A dictionary of helper functions specific to the agent.
        """
        self.llm = llm
        self.prompt = prompt
        self.tools = tools
        self.data_description = data_description
        self.dataset = dataset
        self.helper_functions = helper_functions or {}

    def add_helper_function(self, name, func):
        """
        Add a helper function specific to this agent.

        Parameters:
        - name (str): The name of the function.
        - func (function): The function to be added.
        """
        self.helper_functions[name] = func

    def run(self, question,parameters):
        """
        Run the agent's task using the provided question, available tools, and helper functions.

        Parameters:
        - question (str): The question the agent needs to answer or solve.

        Returns:
        - str: The result of the agent's task.
        """
        # Define the prompt with placeholders for variables
        prompt_temp = ChatPromptTemplate.from_messages(
            [
                ("system", self.prompt.strip()),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )


        print("Parameter for invoking Insights Agent")
        for k,v in parameters.items():
            print(k,v,sep=" : ")
        result = llm.invoke(prompt_temp.invoke({"data_description": self.data_description,
                                                "question": question+f"Use these parameters to filter data: {parameters}",
                                                "messages": [HumanMessage(content=question+"Also, include a single line summary about the parameters in your answer.")]}))

        return result

    def generate_response(self, question,parameters):
        """
        Generate a response using the agent's prompt and data description.

        Parameters:
        - question (str): The question to be answered.

        Returns:
        - str: The generated response based on the prompt and dataset.
        """
        result = self.run(question,parameters)
        response = self.helper_functions['execute_analysis'].invoke(
            {"df": self.dataset, "response_text": result.content})
        return response
    
    def __repr__(self):
        """
        String representation of the agent, showing essential properties.
        """
        return f"Agent(prompt={self.prompt}, tools={self.tools}, data_description={self.data_description}, dataset={self.dataset.head()})"

