## Setting up API key and environment related imports

import os
import re
import uuid

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt


from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage

from langchain.schema import HumanMessage
from langchain_experimental.agents import create_pandas_dataframe_agent

from config import display_saved_plot
from src.core.order_consolidation.dynamic_consolidation import (get_filtered_data,
                                                                get_parameters_values,
                                                                consolidate_shipments,
                                                                calculate_metrics,
                                                                analyze_consolidation_distribution, agent_wrapper)

from src.core.order_consolidation.consolidation_ui import timed_call



PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)


class AgenticCostOptimizer:
    def __init__(self, llm, parameters):
        """
        Initialize the Agentic Cost Optimizer.

        :param llm: The LLM model to use for queries.
        :param parameters: Dictionary containing required parameters.
        """
        self.llm = llm
        self.parameters = parameters
        self.df = parameters.get("df", pd.DataFrame())
        # self.shipment_window_range = (1, 10)
        # self.total_shipment_capacity = 36
        # self.utilization_threshold = 95

    def load_data(self):
        complete_input = os.path.join(os.getcwd() , "src/data/Complete Input.xlsx")
        rate_card_ambient = pd.read_excel(complete_input, sheet_name='AMBIENT')
        rate_card_ambcontrol = pd.read_excel(complete_input, sheet_name='AMBCONTROL')
        return {"rate_card_ambient": rate_card_ambient, "rate_card_ambcontrol": rate_card_ambcontrol}


    def get_filtered_df_from_question(self):
        """Extracts filtered data based on user query parameters."""
        group_field = 'SHORT_POSTCODE' if self.parameters['group_method'] == 'Post Code Level' else 'NAME'
        df = self.parameters['df']
        df['SHIPPED_DATE'] = pd.to_datetime(df['SHIPPED_DATE'], dayfirst=True)

        df = get_filtered_data(self.parameters, df)
        if df.empty:
            raise ValueError("No data available for selected parameters. Try again!")
        return df

    def get_cost_saving_data(self):
        """Runs cost-saving algorithm and returns result DataFrame."""
        
        df = self.get_filtered_df_from_question()
        group_field = 'SHORT_POSTCODE' if self.parameters['group_method'] == 'Post Code Level' else 'NAME'
        st.toast(f"Shape of original data after filtering: {df.shape}")
        
        df['GROUP'] = df['SHORT_POSTCODE' if self.parameters['group_method'] == 'Post Code Level' else 'NAME']
        grouped = df.groupby(['PROD TYPE', 'GROUP'])
        date_range = pd.date_range(start=self.parameters['start_date'], end=self.parameters['end_date'])

        
        best_metrics=None
        best_consolidated_shipments=None
        best_params = None

        all_results = []
        rate_card = self.load_data()
        for shipment_window in range(self.parameters["shipment_window_range"][0], self.parameters["shipment_window_range"][1] + 1):
            # st.write(f"Consolidating orders for shipment window: {shipment_window}")
            # print(f"Consolidating orders for shipment window: {shipment_window}")
            st.toast(f"Consolidating orders for shipment window: {shipment_window}")
            high_priority_limit = 0
            all_consolidated_shipments = []
            for _, group_df in grouped:
                consolidated_shipments, _ = consolidate_shipments(
                    group_df, high_priority_limit, self.parameters["utilization_threshold"], shipment_window, date_range, lambda: None, self.parameters["total_shipment_capacity"],rate_card
                )
                all_consolidated_shipments.extend(consolidated_shipments)
            
            metrics = calculate_metrics(all_consolidated_shipments, df)
            distribution, distribution_percentage = analyze_consolidation_distribution(all_consolidated_shipments, df)
            
            result = {
                'Shipment Window': shipment_window,
                'Total Orders': metrics['Total Orders'],
                'Total Shipments': metrics['Total Shipments'],
                'Total Shipment Cost': round(metrics['Total Shipment Cost'], 1),
                'Total Baseline Cost': round(metrics['Total Baseline Cost'], 1),
                'Cost Savings': metrics['Cost Savings'],
                'Percent Savings': round(metrics['Percent Savings'], 1),
                'Average Utilization': round(metrics['Average Utilization'], 1),
                'CO2 Emission': round(metrics['CO2 Emission'], 1)
            }
            all_results.append(result)

        # Update best results if current combination is better
        if best_metrics is None or metrics['Cost Savings'] > best_metrics['Cost Savings']:
            best_metrics = metrics
            best_consolidated_shipments = all_consolidated_shipments
            best_params = (shipment_window, high_priority_limit, self.parameters["utilization_threshold"])

        # Updating the parameters with adding shipment window vs cost saving table..    
        self.parameters['all_results'] = pd.DataFrame(all_results)
        self.parameters['best_params'] = best_params

    ############################################################################################################################################

    def consolidate_for_shipment_window(self):
        """Runs consolidation algorithm based on the selected shipment window."""
        df = self.get_filtered_df_from_question()
        df['GROUP'] = df['SHORT_POSTCODE' if self.parameters['group_method'] == 'Post Code Level' else 'NAME']
        grouped = df.groupby(['PROD TYPE', 'GROUP'])
        date_range = pd.date_range(start=self.parameters['start_date'], end=self.parameters['end_date'])

        rate_card = self.load_data()
        all_consolidated_shipments = []
        for _, group_df in grouped:
            consolidated_shipments, _ = consolidate_shipments(
                group_df, 0, 95, self.parameters['window'], date_range, lambda: None, self.parameters["total_shipment_capacity"],
                rate_card
            )
            all_consolidated_shipments.extend(consolidated_shipments)

        selected_postcodes = ", ".join(self.parameters["selected_postcodes"]) if self.parameters[
            "selected_postcodes"] else "All Postcodes"
        selected_customers = ", ".join(self.parameters["selected_customers"]) if self.parameters[
            "selected_customers"] else "All Customers"

        metrics = calculate_metrics(all_consolidated_shipments, df)

        self.parameters['all_consolidated_shipments'] = pd.DataFrame(all_consolidated_shipments)
        self.parameters['metrics'] = metrics
        self.parameters['filtered_df'] = df

    def compare_before_and_after_consolidation(self):
        """Compares shipments before and after consolidation."""
        consolidated_df = self.parameters['all_consolidated_shipments']
        df = self.get_filtered_df_from_question()

        before = {
            "Days": df['SHIPPED_DATE'].nunique(),
            "Pallets Per Day": df['Total Pallets'].sum() / df['SHIPPED_DATE'].nunique(),
            "Pallets per Shipment": df['Total Pallets'].sum() / len(df)
        }
        after = {
            "Days": consolidated_df['Date'].nunique(),
            "Pallets Per Day": consolidated_df['Total Pallets'].sum() / consolidated_df['Date'].nunique(),
            "Pallets per Shipment": consolidated_df['Total Pallets'].sum() / len(consolidated_df)
        }

        percentage_change = {
            key: round(((after[key] - before[key]) / before[key]) * 100, 2) for key in before
        }

        comparison_df = pd.DataFrame({"Before": before, "After": after, "% Change": percentage_change})
        self.parameters["comparison_df"] = comparison_df

    def handle_query(self, question,state_parameters):
        """Handles user queries dynamically with conversation history and data processing."""

        def run_agent_query(agent, query, dataframe, max_attempts=3):
            """Runs an agent query with up to `max_attempts` retries on failure.

            Args:
                agent: The agent to invoke.
                query (str): The query to pass to the agent.
                dataframe (pd.DataFrame): DataFrame for response context.
                max_attempts (int, optional): Maximum retry attempts. Defaults to 3.

            Returns:
                str: Final answer or error message after attempts.
            """
            attempt = 0
            while attempt < max_attempts:
                try:
                    # st.info(f"Attempt {attempt + 1} of {max_attempts}...")
                    response = agent.invoke(query)
                    response_ = agent_wrapper(response, dataframe)

                    return response_["final_answer"]

                except Exception as e:
                    attempt += 1
                    # st.warning(f"Error on attempt {attempt}: {e}")

                    if attempt == max_attempts:
                        st.error(f"Failed after {max_attempts} attempts. Please revise the query or check the data.")
                        return f"Error: {e}"

        def display_agent_steps(steps):
            """Displays agent reasoning steps and associated plots."""
            for i, step in enumerate(steps):
                st.write(step['message'])
                for plot_path in step['plot_paths']:
                    display_saved_plot(plot_path)

        chat_history = []
        chat_history.append({"Human": question})

        # Extract parameters from question


        extracted_params = state_parameters
        
        # extracted_params = get_parameters_values(self.parameters["api_key"], question,attempt=0)
        self.parameters.update(extracted_params)
        print("Extracted Parameters:\n")
        for k,v in extracted_params.items():
            print(k,v)

        chat_history.append({"Agent": f"Parameters extracted: {extracted_params}"})

        st.info("Running order-consolidation algorithm...")
        self.get_cost_saving_data()
        # Identify row with maximum cost savings
        max_savings_row = self.parameters['all_results'].loc[
            self.parameters['all_results']['Cost Savings'].idxmax()
        ].to_dict()
        chat_history.append({"Agent": f"Optimum results: {max_savings_row}"})

        agent = create_pandas_dataframe_agent(
            self.llm, self.parameters['all_results'],
            verbose=False, allow_dangerous_code=True,
            handle_parsing_errors=True, return_intermediate_steps=True
        )
        with st.expander("**CORRELATION BETWEEN SHIPMENT WINDOW AND KEY METRICS**" ,expanded=False):
            shipment_query = (
                "Share a quick insights by comparing Shipment Window against Total Shipments, Cost Savings and Total Shipment costs.",
                "The insight should provide overview about how shipment window varies with these factors and which is the best shipment to chose from.",
                "Avoid plots as plot is already there showing the trend, just provide a single or multi-line comment for each comparison.",
                "Use `python_ast_repl_tool` to write a python script and  then print the results in order to pass it to final response.")

            final_answer = run_agent_query(agent, shipment_query, self.parameters['all_results'], max_attempts=3)

        self.parameters["shipment_window_vs_saving_agent_msg"] = final_answer
        chat_history.extend([{"Human": shipment_query}, {"Agent": final_answer}])

        # Determine shipment window
        user_window = None  # Replace with user input logic if needed
        self.parameters["window"] = int(user_window) if user_window else max_savings_row['Shipment Window']

        self.consolidate_for_shipment_window()

        self.compare_before_and_after_consolidation()
        comparison_results = self.parameters["comparison_df"].to_dict()
        chat_history.append({"Agent": f"Comparison results: {comparison_results}"})
        chat = []
        for msg in chat_history:
            key, value = list(msg.items())[0]
            if "Agent" in key:
                if type(value) is not str:
                    value = str(value)
                chat.append(AIMessage(content=value))
            else:
                chat.append(HumanMessage(content=value))

        # result = self.llm.invoke(f"This is the response provided by the Cost Optimization Agent: {chat}. Generate a final response to be shown. Keep an info of all the extracted parameters too.")
        # self.parameters['final_response'] = """Results from "Dynamic Cost Optimization Agent":\n"""+result.content

        result = self.llm.invoke(
                f"""This is the response provided by the Dynamic Cost Optimization Agent: {chat}. 
                Generate a final response to be shown to the user. 
                - Show optimum results.
                - Show comparison results.
                - List all extracted parameters separately and show in a single sentence.
                - Keep the tone professional and clear.
                """
            )
        # - Ensure the response explicitly mentions the agent's name ("Dynamic Cost Optimization Agent").

        # Extracting parameters from `self.parameters`
        parameters_summary = "\n".join([f"- {key}: {value}" for key, value in extracted_params.items()])

        # Structuring the final response
        self.parameters['final_response'] = f"""
        {result.content}
        """


        print("Reponse END from Dynamic Agent")

        return self.parameters

