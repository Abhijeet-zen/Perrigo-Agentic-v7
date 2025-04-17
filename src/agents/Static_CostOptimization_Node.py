## Setting up API key and environment related imports

from dotenv import load_dotenv, find_dotenv
import streamlit as st
from src.core.order_consolidation.static_consolidation import cost_of_columns,consolidations_day_mapping,consolidate_shipments
from src.core.order_consolidation.dynamic_consolidation import get_parameters_values,get_filtered_data
from src.core.order_consolidation.consolidation_ui import timed_call

_ = load_dotenv(find_dotenv())

import pandas as pd
import warnings

warnings.filterwarnings("ignore")

from langchain_core.messages import  AIMessage
from langchain.schema import HumanMessage


class Static_CostOptimization_Class():
    def __init__(self, llm, parameters):
        """
        Initialize the Agentic Cost Optimizer.

        :param llm: The LLM model to use for queries.
        :param parameters: Dictionary containing required parameters.
        """
        self.llm = llm
        self.parameters = parameters

    def find_cost_savings(self):

        scenarios = {
            "5 days delivery scenario": [
                "Mon_Tue_Wed_Thu_Fri", "Mon_Tue_Wed_Thu_Sat", "Mon_Tue_Wed_Thu_Sun",
                "Mon_Tue_Wed_Fri_Sat", "Mon_Tue_Wed_Fri_Sun", "Mon_Tue_Thu_Fri_Sat",
                "Mon_Tue_Thu_Fri_Sun", "Mon_Wed_Thu_Fri_Sat", "Mon_Wed_Thu_Fri_Sun",
                "Tue_Wed_Thu_Fri_Sat", "Tue_Wed_Thu_Fri_Sun"
            ],
            "4 days delivery scenario": [
                "Mon_Tue_Wed_Thu", "Mon_Tue_Wed_Fri", "Mon_Tue_Wed_Sat", "Mon_Tue_Wed_Sun",
                "Mon_Tue_Thu_Fri", "Mon_Tue_Thu_Sat", "Mon_Tue_Thu_Sun", "Mon_Wed_Thu_Fri",
                "Mon_Wed_Thu_Sat", "Mon_Wed_Thu_Sun", "Tue_Wed_Thu_Fri", "Tue_Wed_Thu_Sat",
                "Tue_Wed_Thu_Sun"
            ],
            "3 days delivery scenario": [
                "Mon_Tue_Wed", "Mon_Tue_Thu", "Mon_Tue_Fri", "Mon_Tue_Sat", "Mon_Tue_Sun",
                "Mon_Wed_Thu", "Mon_Wed_Fri", "Mon_Wed_Sat", "Mon_Wed_Sun",
                "Tue_Wed_Thu", "Tue_Wed_Fri", "Tue_Wed_Sat", "Tue_Wed_Sun"
            ],
            "2 days delivery scenario": [
                "Mon_Tue", "Mon_Wed", "Mon_Thu", "Mon_Fri", "Mon_Sat", "Mon_Sun",
                "Tue_Wed", "Tue_Thu", "Tue_Fri", "Tue_Sat", "Tue_Sun",
                "Wed_Thu", "Wed_Fri", "Wed_Sat", "Wed_Sun"
            ],
            "1 day delivery scenario": [
                "Only_Mon", "Only_Tue", "Only_Wed", "Only_Thu", "Only_Fri", "Only_Sat", "Only_Sun"
            ]
        }

        scenarios = scenarios if self.parameters["scenario"] is None else {self.parameters["scenario"]:scenarios[self.parameters["scenario"]]}
        filter_data = get_filtered_data(self.parameters, self.parameters['complete_input'])

        aggregated_data, total_shipment_cost = cost_of_columns(filter_data, self.parameters['rate_card'])

        all_results = pd.DataFrame()
        best_consolidated_df = pd.DataFrame()
        optimal_consolidation_cost = 9999999
        for k, v in scenarios.items():
            st.toast(f"Running cost saving for {k}.")
            days = k
            scene = v
            scenario_results = []
            for scenario in scene:
                day_mapping = consolidations_day_mapping[scenario]
                consolidated_data, total_consolidated_cost = consolidate_shipments(aggregated_data, self.parameters['rate_card'],
                                                                                   day_mapping)

                scenario_results.append({
                    'days': days,
                    'scenario': scenario,
                    'total_consolidated_cost': total_consolidated_cost,
                    'num_shipments': len(consolidated_data.index),
                    'avg_pallets': round(consolidated_data['Total Pallets'].mean(), 2)
                })
                if total_consolidated_cost<optimal_consolidation_cost:
                    optimal_consolidation_cost = total_consolidated_cost
                    best_consolidated_df = consolidated_data
            all_results = pd.concat([all_results, pd.DataFrame(scenario_results)])

        sorted_results = all_results.sort_values(by='total_consolidated_cost', ascending=True)
        # print("sorted results",sorted_results)
        best_scenario = sorted_results.iloc[0].to_dict()
        # st.dataframe(sorted_results)
        all_results.reset_index(inplace=True, drop=True)
        self.parameters["filtered_df"] = filter_data
        self.parameters['all_results'] = all_results
        self.parameters['best_scenario'] = best_scenario
        self.parameters["aggregated_data"] = aggregated_data
        self.parameters["total_shipment_cost"] = total_shipment_cost
        self.parameters["consolidated_data"] = best_consolidated_df

    def handle_query(self, question,state_parameters):
        chat_history = [{"Human": question}]

        st.info("Extracting parameters from question...")
        # extracted_params = get_parameters_values(self.parameters["api_key"], question,attempt=0)

        extracted_params = state_parameters
        

        print("Extracted Parameters:\n")
        for k,v in extracted_params.items():
            print(k,v)
        self.parameters.update(extracted_params)
        chat_history.append({"Agent": f"Parameters extracted: {extracted_params}"})

        st.info("Running order-consolidation algorithm...")
        self.find_cost_savings()

        chat_history.append({"Agent": f"Scenarios of all possible days: {self.parameters['all_results']}"})
        chat_history.append({"Agent": f"Best scenarios for cost savings: {self.parameters['best_scenario']}"})

        chat = []
        for msg in chat_history:
            key, value = list(msg.items())[0]
            if "Agent" in key:
                if type(value) is not str:
                    value = str(value)
                chat.append(AIMessage(content=value))
            else:
                chat.append(HumanMessage(content=value))

        st.info("Generating final answer...")
        # result = self.llm.invoke(f"This is the response provided by the Static Cost Optimization Agent: {chat}. Generate a final response to be shown. Keep an info of all the extracted parameters too.")
        # self.parameters['final_response'] = """Results from "Static Cost Optimization Agent":\n"""+result.content


        result = self.llm.invoke(
            f"""This is the response provided by the Static Cost Optimization Agent: {chat}. 
            Generate a final response to be shown to the user. 
            - Show best scenario results for cost savings.
            - List all extracted parameters separately and show in a single sentence.
            - Keep the tone professional and clear.
            """)
        # - Ensure the response explicitly mentions the agent's name ("Static Cost Optimization Agent").
        # Extracting parameters from `self.parameters`
        parameters_summary = "\n".join([f"- {key}: {value}" for key, value in extracted_params.items()])

        # Structuring the final response
        self.parameters['final_response'] = f"""
        {result.content}  
        """

        print("Reponse END from Static Agent")

        return self.parameters








