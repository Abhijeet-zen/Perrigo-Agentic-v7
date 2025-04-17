import os
import re
import openai
import streamlit as st
import getpass
import random
import ast
import time
import datetime
from itertools import product
from collections import Counter, defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyecharts import options as opts
from pyecharts.charts import Calendar
from pyecharts.globals import ThemeType
from pyecharts.commons.utils import JsCode
import streamlit.components.v1 as components
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prompt_templates import load_template


def execute_plot_code(plot_code: str, df) -> list:
    """
    Executes the provided plotting code. It searches for every occurrence of plt.show() in the code and
    replaces it with a plt.savefig() call that saves the current figure to a unique file.
    If no plt.show() is present and no plt.savefig() exists, it appends a plt.savefig() call at the end.

    Args:
        plot_code (str): The code to generate one or more plots.

    Returns:
        list: A list of file paths where the plots were saved.
    """
    plot_paths = []

    # Function to replace each plt.show() with a unique plt.savefig() call
    def replace_show(match):
        new_path = os.path.join(PLOT_DIR, f"plot_{uuid.uuid4().hex}.png")
        plot_paths.append(new_path)
        # Ensure proper indentation is preserved (using the match group 1)
        indent = match.group(1) if match.lastindex and match.group(1) else ""
        return f"{indent}plt.savefig('{new_path}', bbox_inches='tight')"

    # Replace all occurrences of plt.show() with unique plt.savefig() calls
    sanitized_code = re.sub(r'(^\s*)plt\.show\(\)', replace_show, plot_code, flags=re.MULTILINE)

    # If no plt.show was found and no plt.savefig exists in the code, append one at the end
    if not re.search(r"plt\.savefig", sanitized_code):
        new_path = os.path.join(PLOT_DIR, f"plot_{uuid.uuid4().hex}.png")
        sanitized_code += f"\nplt.savefig('{new_path}', bbox_inches='tight')"
        plot_paths.append(new_path)

    #     print("Sanitized Code:\n", sanitized_code)

    exec_globals = {"df": df, "sns": sns, "plt": plt}
    try:
        exec(sanitized_code, exec_globals)
        plt.close('all')
    except Exception as e:
        return [f"Error generating plot: {e}"]

    return plot_paths


def extract_plot_code(intermediate_steps: list) -> tuple:
    """
    Extracts the plotting code from the agent's intermediate steps.

    Args:
        intermediate_steps (list): Intermediate steps from the agent response.

    Returns:
        tuple: (plot_code, response, thought)
    """
    for step in intermediate_steps:
        artifacts, _ = step

        # The agent's intermediate steps may contain the code under a key like 'tool_input'
        tool_input_ = artifacts.tool_input
        agent_message = artifacts.log

        # Extract plot code (everything after "```python" and before "```")
        match = re.search(r"```python(.*?)```", tool_input_, re.DOTALL)
        plot_code = match.group(1).strip() if match else None

        # Extract message (everything before "Thought:")
        response_match = re.search(r'^(.*?)\s*Thought:', agent_message, re.DOTALL)
        response = response_match.group(1).strip() if response_match else agent_message.strip()

        # Extract thought (text between "Thought:" and "Action:")
        thought_match = re.search(r'Thought:\s*(.*?)\s*Action:', agent_message, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""

    return plot_code, response, thought


def agent_wrapper(agent_response: dict, df) -> dict:
    """
    Wraps the agent response to extract, execute, and display plotting code for each intermediate step.
    For each step, any generated plots are saved using unique file names.

    The final output is constructed to show:
      - Step 1 message
      - Step 1 plot paths
      - Step 2 message
      - Step 2 plot paths
      - ...
      - Final agent response

    Args:
        agent_response (dict): Response from the agent.

    Returns:
        dict: Contains the agent input, a list of step outputs (each with a message and plot paths),
              and a final_answer string combining all.
    """
    intermediate_steps = agent_response.get("intermediate_steps", [])
    step_outputs = []

    for step in intermediate_steps:
        artifacts, _ = step
        tool_input_ = artifacts.tool_input
        agent_log = artifacts.log

        # Extract the plotting code from the tool_input
        match = re.search(r"```python(.*?)```", tool_input_, re.DOTALL)
        plot_code = match.group(1).strip() if match else None
        plot_code = plot_code if "plt.show" in plot_code else None

        # Extract the message (everything before "Thought:") and optional thought
        message_match = re.search(r'^(.*?)\s*Thought:', agent_log, re.DOTALL)
        message = message_match.group(1).strip() if message_match else agent_log.strip()
        thought_match = re.search(r'Thought:\s*(.*?)\s*Action:', agent_log, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""
        full_message = message + ("\n" + thought if thought else "")

        # Execute the plotting code and get a list of plot paths
        plot_paths = execute_plot_code(plot_code, df) if plot_code else []

        step_outputs.append({
            "message": full_message,
            "plot_paths": plot_paths
        })

    # Build the final answer by interleaving messages and the list of plot paths
    final_message = ""
    for idx, step in enumerate(step_outputs, 1):
        final_message += f"Step {idx} Message:\n{step['message']}\n"
        if step['plot_paths']:
            for i, path in enumerate(step['plot_paths'], 1):
                final_message += f"Step {idx} Plot {i}: {path}\n"
        else:
            final_message += f"Step {idx} Plot: No plot generated.\n"

    final_agent_response = agent_response.get("output", "")
    #     final_message += "\nFinal Agent Response:\n" + final_agent_response

    return {
        "input": agent_response.get("input"),
        "steps": step_outputs,
        "final_answer": final_agent_response
    }


def create_consolidated_shipments_calendar_dynamic(consolidated_df):
    # Group by Date and calculate both Shipments Count and Total Orders
    df_consolidated = consolidated_df.groupby('Date').agg({
        'Orders': ['count', lambda x: sum(len(orders) for orders in x)]
    }).reset_index()
    df_consolidated.columns = ['Date', 'Shipments Count', 'Orders Count']

    # Split data by year
    df_2023 = df_consolidated[df_consolidated['Date'].dt.year == 2023]
    df_2024 = df_consolidated[df_consolidated['Date'].dt.year == 2024]
    df_2025 = df_consolidated[df_consolidated['Date'].dt.year == 2025]

    calendar_data_2023 = df_2023[['Date', 'Shipments Count', 'Orders Count']].values.tolist()
    calendar_data_2024 = df_2024[['Date', 'Shipments Count', 'Orders Count']].values.tolist()
    calendar_data_2025 = df_2025[['Date', 'Shipments Count', 'Orders Count']].values.tolist()

    def create_calendar(data, year):
        return (
            Calendar(init_opts=opts.InitOpts(width="984px", height="200px", theme=ThemeType.ROMANTIC))
            .add(
                series_name="",
                yaxis_data=data,
                calendar_opts=opts.CalendarOpts(
                    pos_top="50",
                    pos_left="40",
                    pos_right="30",
                    range_=str(year),
                    yearlabel_opts=opts.CalendarYearLabelOpts(is_show=False),
                    daylabel_opts=opts.CalendarDayLabelOpts(name_map=['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']),
                    monthlabel_opts=opts.CalendarMonthLabelOpts(name_map="en"),
                ),
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title=f"Calendar Heatmap for Orders and Shipments After Consolidation ({year})"),
                visualmap_opts=opts.VisualMapOpts(
                    max_=max(item[2] for item in data) if data else 0,
                    min_=min(item[2] for item in data) if data else 0,
                    orient="horizontal",
                    is_piecewise=False,
                    pos_bottom="20",
                    pos_left="center",
                    range_color=["#E8F5E9", "#1B5E20"],
                    is_show=False,
                ),
                tooltip_opts=opts.TooltipOpts(
                    formatter=JsCode(
                        """
                        function (p) {
                            var date = new Date(p.data[0]);
                            var day = date.getDate().toString().padStart(2, '0');
                            var month = (date.getMonth() + 1).toString().padStart(2, '0');
                            var year = date.getFullYear();
                            return 'Date: ' + day + '/' + month + '/' + year + 
                                   '<br/>Orders: ' + p.data[2] +
                                   '<br/>Shipments: ' + p.data[1];
                        }
                        """
                    )
                )
            )
        )

    calendar_2023 = create_calendar(calendar_data_2023, 2023)
    calendar_2024 = create_calendar(calendar_data_2024, 2024)
    calendar_2025 = create_calendar(calendar_data_2025, 2025)

    return calendar_2023, calendar_2024, calendar_2025


def create_original_orders_calendar_dynamic(original_df):
    df_original = original_df.groupby('SHIPPED_DATE').size().reset_index(name='Orders Shipped')

    # Split data by year
    df_2023 = df_original[df_original['SHIPPED_DATE'].dt.year == 2023]
    df_2024 = df_original[df_original['SHIPPED_DATE'].dt.year == 2024]
    df_2025 = df_original[df_original['SHIPPED_DATE'].dt.year == 2025]

    calendar_data_2023 = df_2023[['SHIPPED_DATE', 'Orders Shipped']].values.tolist()
    calendar_data_2024 = df_2024[['SHIPPED_DATE', 'Orders Shipped']].values.tolist()
    calendar_data_2025 = df_2025[['SHIPPED_DATE', 'Orders Shipped']].values.tolist()

    def create_calendar(data, year):
        return (
            Calendar(init_opts=opts.InitOpts(width="984px", height="200px", theme=ThemeType.ROMANTIC))
            .add(
                series_name="",
                yaxis_data=data,
                calendar_opts=opts.CalendarOpts(
                    pos_top="50",
                    pos_left="40",
                    pos_right="30",
                    range_=str(year),
                    yearlabel_opts=opts.CalendarYearLabelOpts(is_show=False),
                    daylabel_opts=opts.CalendarDayLabelOpts(name_map=['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']),
                    monthlabel_opts=opts.CalendarMonthLabelOpts(name_map="en"),
                ),
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"Calendar Heatmap for Orders Shipped Before Consolidation ({year})"),
                visualmap_opts=opts.VisualMapOpts(
                    max_=max(item[1] for item in data) if data else 0,
                    min_=min(item[1] for item in data) if data else 0,
                    orient="horizontal",
                    is_piecewise=False,
                    pos_bottom="20",
                    pos_left="center",
                    range_color=["#E8F5E9", "#1B5E20"],
                    is_show=False,
                ),
                tooltip_opts=opts.TooltipOpts(
                    formatter=JsCode(
                        """
                        function (p) {
                            var date = new Date(p.data[0]);
                            var day = date.getDate().toString().padStart(2, '0');
                            var month = (date.getMonth() + 1).toString().padStart(2, '0');
                            var year = date.getFullYear();
                            return 'Date: ' + day + '/' + month + '/' + year + '<br/>Orders: ' + p.data[1];
                        }
                        """
                    )
                )
            )
        )

    calendar_2023 = create_calendar(calendar_data_2023, 2023)
    calendar_2024 = create_calendar(calendar_data_2024, 2024)
    calendar_2025 = create_calendar(calendar_data_2025, 2025)

    return calendar_2023, calendar_2024, calendar_2025


def create_heatmap_and_bar_charts_dynamic(consolidated_df, original_df, start_date, end_date):
    # Create calendar charts (existing code)
    chart_original_2023, chart_original_2024, chart_original_2025 = create_original_orders_calendar_dynamic(original_df)
    chart_consolidated_2023, chart_consolidated_2024, chart_consolidated_2025 = create_consolidated_shipments_calendar_dynamic(
        consolidated_df)

    # Create bar charts for orders over time
    def create_bar_charts(df_original, df_consolidated, year):
        # Filter data for the specific year
        mask_original = df_original['SHIPPED_DATE'].dt.year == year
        year_data_original = df_original[mask_original]

        # For consolidated data
        if 'Date' in df_consolidated.columns:
            mask_consolidated = pd.to_datetime(df_consolidated['Date']).dt.year == year
            year_data_consolidated = df_consolidated[mask_consolidated]
        else:
            year_data_consolidated = pd.DataFrame()

        # Create subplot figure with shared x-axis
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=(
                f'Daily Orders Before Consolidation ({year})',
                f'Daily Orders After Consolidation ({year})'
            )
        )

        # Add bar chart for original orders
        if not year_data_original.empty:
            daily_orders = year_data_original.groupby('SHIPPED_DATE').size().reset_index()
            daily_orders.columns = ['Date', 'Orders']

            fig.add_trace(
                go.Bar(
                    x=daily_orders['Date'],
                    y=daily_orders['Orders'],
                    name='Orders',
                    marker_color='#1f77b4'
                ),
                row=1,
                col=1
            )

        # Add bar chart for consolidated orders
        if not year_data_consolidated.empty:
            daily_consolidated = year_data_consolidated.groupby('Date').agg({
                'Orders': lambda x: sum(len(orders) for orders in x)
            }).reset_index()

            fig.add_trace(
                go.Bar(
                    x=daily_consolidated['Date'],
                    y=daily_consolidated['Orders'],
                    name='Orders',
                    marker_color='#749f77'
                ),
                row=2,
                col=1
            )

        # Update layout
        fig.update_layout(
            height=500,  # Increased height for better visibility
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=20, t=60, b=20),
            hovermode='x unified'
        )

        # Update x-axes
        fig.update_xaxes(
            rangeslider=dict(
                visible=True,
                thickness=0.05,  # Make the rangeslider thinner
                bgcolor='#F4F4F4',  # Light gray background
                bordercolor='#DEDEDE',  # Slightly darker border
            ),
            row=2,
            col=1
        )
        fig.update_xaxes(
            rangeslider=dict(visible=False),
            row=1,
            col=1
        )

        # Update y-axes
        fig.update_yaxes(title_text="Number of Orders", row=1, col=1)
        fig.update_yaxes(title_text="Number of Orders", row=2, col=1)

        return fig

    # Create bar charts for both years
    bar_charts_2023 = create_bar_charts(original_df, consolidated_df, 2023)
    bar_charts_2024 = create_bar_charts(original_df, consolidated_df, 2024)
    bar_charts_2025 = create_bar_charts(original_df, consolidated_df, 2025)

    return {
        2023: (chart_original_2023, chart_consolidated_2023, bar_charts_2023),
        2024: (chart_original_2024, chart_consolidated_2024, bar_charts_2024),
        2025: (chart_original_2025, chart_consolidated_2025, bar_charts_2025)
    }



def get_chatgpt_response(api_key, instructions, user_query):
    """
    Sends a query to OpenAI's ChatCompletion API with the given instructions and user query.
    """
    # Set the API key
    # openai.api_key = api_key

    try:
        # Send the query to OpenAI ChatCompletion API
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # Specify the GPT-4 model
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": user_query}
            ],
            max_tokens=500,  # Adjust token limit based on your needs
            temperature=0.0  # Adjust for creativity (0.7 is a balanced value)
        )
        # Extract and return the assistant's response
        return response.choices[0].message.content

    except openai.OpenAIError as e:
        # Handle OpenAI-specific errors
        return f"An error occurred with the OpenAI API: {str(e)}"

    except Exception as e:
        # Handle other exceptions (e.g., network issues)
        return f"An unexpected error occurred: {str(e)}"



def ask_openai(selected_customers, selected_postcodes, customers, postcodes,query):
    """
    Sends a question and data context to OpenAI API for processing.
    """
    # Formulate the prompt
    prompt_template = load_template("customer_postcode_matching_prompt.txt")
    prompt = prompt_template.format(
        selected_customers=selected_customers,
        selected_postcodes=selected_postcodes,
        customers=customers,
        postcodes=postcodes,
        query=query
    )

    # Call OpenAI API
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system",
             "content": "You are an assistant skilled at answering questions about searching something"},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
        temperature=0.0
    )
    return response.choices[0].message.content


def get_parameters_values(api_key, query,attempt):
    if attempt<3:
        instructions = load_template("cost_parameters_prompt.txt")
        # print("Question:\n",query)
        # print("-"*30)
        # print(instructions)
        # print("-"*30)
        response = get_chatgpt_response(api_key, instructions, query)
        # print(response)
        if response:
            try:
                extracted_code = eval(response)
                # Load input file and get customers and postcodes
                input_data = pd.read_excel(os.path.join(os.getcwd() , "src/data/Complete Input.xlsx"))
                customers = input_data["NAME"].unique()
                postcodes = input_data["SHORT_POSTCODE"].unique()
                
                # Extract initial selected values from response
                selected_customers = extracted_code.get('selected_customers', [])
                selected_postcodes = extracted_code.get('selected_postcodes', [])
                
                # Ask further OpenAI for matching customers and postcodes
                answer = ask_openai(selected_customers, selected_postcodes, customers, postcodes,query)
                
                # Extract matched_customers from answer
                customers_match = re.search(r"matched_customers:\s*(\[.*?\])", answer)
                matched_customers = ast.literal_eval(customers_match.group(1)) if customers_match else []
                
                # Extract matched_postcodes from answer
                postcodes_match = re.search(r"matched_postcodes:\s*(\[.*?\])", answer)
                matched_postcodes = ast.literal_eval(postcodes_match.group(1)) if postcodes_match else []
                
                print(matched_customers)
                print(matched_postcodes)
                extracted_code['selected_customers'] = matched_customers
                extracted_code['selected_postcodes'] = matched_postcodes


                # Ensure default values for new parameters if not extracted
                if 'scenario' not in extracted_code:
                    extracted_code['scenario'] = None
                if 'shipment_window_range' not in extracted_code:
                    extracted_code['shipment_window_range'] = (1, 10)
                if 'total_shipment_capacity' not in extracted_code:
                    extracted_code['total_shipment_capacity'] = 36
                if 'utilization_threshold' not in extracted_code:
                    extracted_code['utilization_threshold'] = 95
                
                return extracted_code

            except Exception as e:
                    attempt+=1
                    st.error(f"Error while extracting params: {str(e)}")
                    get_parameters_values(api_key, query+"\n"+str(e)+"Fix error and Try again.",attempt)

    else:
        print("Running with default params.")
        # Default parameters in case of an exception
        default_param = {
            "start_date": "2024-01-01",
            "end_date": "2024-03-31",
            "group_method": "Post Code Level",
            "all_post_code": False,
            "all_customers": None,
            "selected_postcodes": ["NG"],
            "selected_customers": [],
            "scenario": None,
            "shipment_window_range": (1, 10),
            "total_shipment_capacity": 36,
            "utilization_threshold": 95
        }
        return default_param


def get_filtered_data(parameters, df):
    global group_field
    global group_method

    group_method = parameters['group_method']
    group_field = 'SHORT_POSTCODE' if group_method == 'Post Code Level' else 'NAME'

    # Month selection
    start_date = parameters['start_date']
    end_date = parameters['end_date']

    # Filter data based on selected date range
    df = df[(df['SHIPPED_DATE'] >= start_date) & (df['SHIPPED_DATE'] <= end_date)]
    # print("only date filter", df.shape) ### checkk

    # Add checkbox and conditional dropdown for selecting post codes or customers

    if group_method == 'Post Code Level':
        all_postcodes = parameters['all_post_code']

        if not all_postcodes:
            selected_postcodes = parameters['selected_postcodes']
            selected_postcodes = [z.strip('') for z in selected_postcodes]
    else:  # Customer Level
        all_customers = parameters['all_customers']
        if not all_customers:
            selected_customers = parameters['selected_customers']
            selected_customers = [c.strip('') for c in selected_customers]
    # Filter the dataframe based on the selection
    if group_method == 'Post Code Level' and not all_postcodes:
        if selected_postcodes:  # Only filter if some postcodes are selected
            df = df[df['SHORT_POSTCODE'].str.strip('').isin(selected_postcodes)]
        else:
            return pd.DataFrame()

    elif group_method == 'Customer Level' and not all_customers:
        if selected_customers:  # Only filter if some customers are selected
            df = df[df['NAME'].str.strip('').isin(selected_customers)]
        else:
            return pd.DataFrame()

    return df


def calculate_metrics(all_consolidated_shipments, df):
    total_orders = sum(len(shipment['Orders']) for shipment in all_consolidated_shipments)
    total_shipments = len(all_consolidated_shipments)
    total_pallets = sum(shipment['Total Pallets'] for shipment in all_consolidated_shipments)
    total_utilization = sum(shipment['Utilization %'] for shipment in all_consolidated_shipments)
    average_utilization = total_utilization / total_shipments if total_shipments > 0 else 0
    total_shipment_cost = sum(
        shipment['Shipment Cost'] for shipment in all_consolidated_shipments if not pd.isna(shipment['Shipment Cost']))
    total_baseline_cost = sum(
        shipment['Baseline Cost'] for shipment in all_consolidated_shipments if not pd.isna(shipment['Baseline Cost']))
    cost_savings = total_baseline_cost - total_shipment_cost
    percent_savings = (cost_savings / total_baseline_cost) * 100 if total_baseline_cost > 0 else 0

    # Calculate CO2 Emission
    total_distance = 0
    sum_dist = 0
    for shipment in all_consolidated_shipments:
        order_ids = shipment['Orders']
        avg_distance = df[df['ORDER_ID'].isin(order_ids)]['Distance'].mean()
        sum_distance = df[df['ORDER_ID'].isin(order_ids)]['Distance'].sum()
        total_distance += avg_distance
        sum_dist += sum_distance
    co2_emission = (sum_dist - total_distance) * 2  # Multiply by 2

    metrics = {
        'Total Orders': total_orders,
        'Total Shipments': total_shipments,
        'Total Pallets': total_pallets,
        'Average Utilization': average_utilization,
        'Total Shipment Cost': total_shipment_cost,
        'Total Baseline Cost': total_baseline_cost,
        'Cost Savings': round(cost_savings, 1),
        'Percent Savings': percent_savings,
        'CO2 Emission': co2_emission
    }

    return metrics


def analyze_consolidation_distribution(all_consolidated_shipments, df):
    distribution = {}
    for shipment in all_consolidated_shipments:
        consolidation_date = shipment['Date']
        for order_id in shipment['Orders']:
            shipped_date = df.loc[df['ORDER_ID'] == order_id, 'SHIPPED_DATE'].iloc[0]
            days_difference = (shipped_date - consolidation_date).days
            if days_difference not in distribution:
                distribution[days_difference] = 0
            distribution[days_difference] += 1

    total_orders = sum(distribution.values())
    distribution_percentage = {k: round((v / total_orders) * 100, 1) for k, v in distribution.items()}
    return distribution, distribution_percentage


def create_utilization_chart(all_consolidated_shipments):
    print(all_consolidated_shipments)  #### checkkkk
    utilization_bins = {f"{i}-{i + 5}%": 0 for i in range(0, 100, 5)}
    for shipment in all_consolidated_shipments:
        utilization = shipment['Utilization %']
        bin_index = min(int(utilization // 5) * 5, 95)  # Cap at 95-100% bin
        bin_key = f"{bin_index}-{bin_index + 5}%"
        utilization_bins[bin_key] += 1

    total_shipments = len(all_consolidated_shipments)
    utilization_distribution = {bin: (count / total_shipments) * 100 for bin, count in utilization_bins.items()}

    fig = go.Figure(data=[go.Bar(x=list(utilization_distribution.keys()), y=list(utilization_distribution.values()),
                                 marker_color='#1f77b4')])
    fig.update_layout(
        title={'text': 'Utilization Distribution', 'font': {'size': 22}},
        xaxis_title='Utilization Range',
        yaxis_title='Percentage of Shipments',
        width=1000,
        height=500
    )

    return fig



def calculate_priority(shipped_date, current_date, shipment_window):
    days_left = (shipped_date - current_date).days
    if 0 <= days_left <= shipment_window:
        return days_left
    return np.nan


def best_fit_decreasing(items, capacity):
    items = sorted(items, key=lambda x: x['Total Pallets'], reverse=True)
    shipments = []

    for item in items:
        best_shipment = None
        min_space = capacity + 1

        for shipment in shipments:
            current_load = sum(order['Total Pallets'] for order in shipment)
            new_load = current_load + item['Total Pallets']

            if new_load <= capacity:
                space_left = capacity - current_load
            else:
                continue  # Skip this shipment if adding the item would exceed capacity

            if item['Total Pallets'] <= space_left < min_space:
                best_shipment = shipment
                min_space = space_left

        if best_shipment:
            best_shipment.append(item)
        else:
            shipments.append([item])

    return shipments


def get_baseline_cost(prod_type, short_postcode, pallets, rate_card):
    total_cost = 0
    for pallet in pallets:
        cost = get_shipment_cost(prod_type, short_postcode, pallet, rate_card)
        if pd.isna(cost):
            return np.nan
        total_cost += cost
    return round(total_cost, 1)


def load_data():
    complete_input = os.path.join(os.getcwd() , "src/data/Complete Input.xlsx")
    rate_card_ambient = pd.read_excel(complete_input, sheet_name='AMBIENT')
    rate_card_ambcontrol = pd.read_excel(complete_input, sheet_name='AMBCONTROL')
    return {"rate_card_ambient": rate_card_ambient, "rate_card_ambcontrol": rate_card_ambcontrol}

def get_shipment_cost(prod_type, short_postcode, total_pallets, rate_card):
    rate_card_ambient, rate_card_ambcontrol = rate_card["rate_card_ambient"], rate_card["rate_card_ambcontrol"]
    if prod_type == 'AMBIENT':
        rate_card = rate_card_ambient
    elif prod_type == 'AMBCONTROL':
        rate_card = rate_card_ambcontrol
    else:
        return np.nan

    row = rate_card[rate_card['SHORT_POSTCODE'] == short_postcode]

    if row.empty:
        return np.nan

    cost_per_pallet = row.get(total_pallets, np.nan).values[0]

    if pd.isna(cost_per_pallet):
        return np.nan

    shipment_cost = round(cost_per_pallet * total_pallets, 1)
    return shipment_cost


def process_shipment(shipment, consolidated_shipments, allocation_matrix, working_df, current_date, capacity,
                     rate_card):
    total_pallets = sum(order['Total Pallets'] for order in shipment)
    utilization = (total_pallets / capacity) * 100

    prod_type = shipment[0]['PROD TYPE']
    short_postcode = shipment[0]['SHORT_POSTCODE']
    shipment_cost = get_shipment_cost(prod_type, short_postcode, total_pallets, rate_card)


    pallets = [order['Total Pallets'] for order in shipment]

    baseline_cost = get_baseline_cost(prod_type, short_postcode, pallets, rate_card)

    shipment_info = {
        'Date': current_date,
        'Orders': [order['ORDER_ID'] for order in shipment],
        'Total Pallets': total_pallets,
        'Capacity': capacity,
        'Utilization %': round(utilization, 1),
        'Order Count': len(shipment),
        'Pallets': pallets,
        'PROD TYPE': prod_type,
        'GROUP': shipment[0]['GROUP'],
        'Shipment Cost': shipment_cost,
        'Baseline Cost': baseline_cost,
        'SHORT_POSTCODE': short_postcode,
        'Load Type': 'Full' if total_pallets > 26 else 'Partial'
    }

    if group_method == 'NAME':
        shipment_info['NAME'] = shipment[0]['NAME']

    consolidated_shipments.append(shipment_info)

    for order in shipment:
        allocation_matrix.loc[order['ORDER_ID'], current_date] = 1
        working_df.drop(working_df[working_df['ORDER_ID'] == order['ORDER_ID']].index, inplace=True)


def consolidate_shipments(df, high_priority_limit, utilization_threshold, shipment_window, date_range,
                          progress_callback, capacity, rate_card):
    consolidated_shipments = []
    allocation_matrix = pd.DataFrame(0, index=df['ORDER_ID'], columns=date_range)

    working_df = df.copy()

    for current_date in date_range:

        working_df.loc[:, 'Priority'] = working_df['SHIPPED_DATE'].apply(
            lambda x: calculate_priority(x, current_date, shipment_window))


        if (working_df['Priority'] == 0).any():
            eligible_orders = working_df[working_df['Priority'].notnull()].sort_values('Priority')
            high_priority_orders = eligible_orders[eligible_orders['Priority'] <= high_priority_limit].to_dict(
                'records')
            low_priority_orders = eligible_orders[eligible_orders['Priority'] > high_priority_limit].to_dict('records')

            if high_priority_orders or low_priority_orders:
                # Process high priority orders first
                high_priority_shipments = best_fit_decreasing(high_priority_orders, capacity)

                # Try to fill high priority shipments with low priority orders
                for shipment in high_priority_shipments:
                    current_load = sum(order['Total Pallets'] for order in shipment)
                    space_left = capacity - current_load  # Use the variable capacity

                    if space_left > 0:
                        low_priority_orders.sort(key=lambda x: x['Total Pallets'], reverse=True)
                        for low_priority_order in low_priority_orders[:]:
                            if low_priority_order['Total Pallets'] <= space_left:
                                shipment.append(low_priority_order)
                                space_left -= low_priority_order['Total Pallets']
                                low_priority_orders.remove(low_priority_order)
                            if space_left == 0:
                                break

                # Process remaining low priority orders
                low_priority_shipments = best_fit_decreasing(low_priority_orders, capacity)

                # Process all shipments
                all_shipments = high_priority_shipments + low_priority_shipments
                for shipment in all_shipments:
                    total_pallets = sum(order['Total Pallets'] for order in shipment)
                    utilization = (total_pallets / capacity) * 100

                    # Always process shipments with high priority orders, apply threshold only to pure low priority shipments
                    if any(order['Priority'] <= high_priority_limit for order in
                           shipment) or utilization >= utilization_threshold:
                        process_shipment(shipment, consolidated_shipments, allocation_matrix, working_df, current_date,
                                         capacity, rate_card)
        progress_callback()

    return consolidated_shipments, allocation_matrix

