import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Image as IPythonImage
from PIL import Image

import time
import datetime
from itertools import product
from collections import Counter, defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
from src.core.order_consolidation.static_consolidation import create_heatmap_and_bar_charts_static
from src.core.order_consolidation.dynamic_consolidation import create_heatmap_and_bar_charts_dynamic


def timed_call(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    exec_time = end_time - start_time
    return result, exec_time


def create_shipment_window_vs_saving_plot(all_results, best_params):
    # Create a dataframe with all simulation results
    results_df = pd.DataFrame(all_results)

    # Preprocess the data to keep only the row with max Cost Savings for each Shipment Window
    optimal_results = results_df.loc[results_df.groupby(['Shipment Window'])['Cost Savings'].idxmax()]

    # Create ColumnDataSource
    source = ColumnDataSource(optimal_results)

    with st.expander("**SHIPMENT WINDOW COMPARISION**", expanded=False):
        shipment_text = (
            f"For each shipment window:\n\n"
            f"- Shipments are grouped together through the consolidation function.\n"
            f"- Key performance metrics, such as cost savings, utilization, and emissions, are calculated.\n"
            f"- The cost savings are compared across different shipment windows to identify the most efficient one.\n"
            f"- On analyzing this data , the best shipment window is observed to be  **{best_params[0]}** days."
        )
        # shipment_rephrase_text = rephrase_text(api_key , shipment_text)
        st.write(shipment_text)

        # Select the best rows for each shipment window
        best_results = results_df.loc[results_df.groupby('Shipment Window')['Percent Savings'].idxmax()]

        # Sort by Shipment Window
        best_results = best_results.sort_values('Shipment Window')

        # Create a complete range of shipment windows from 0 to 30
        all_windows = list(range(0, 31))

        # Create the subplot figure
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add the stacked bar chart
        fig.add_trace(
            go.Bar(
                x=all_windows,
                y=[best_results[best_results['Shipment Window'] == w]['Total Shipment Cost'].values[0] if w in
                                                                                                          best_results[
                                                                                                              'Shipment Window'].values else 0
                   for w in all_windows],
                name='Total Shipment Cost',
                marker_color='#1f77b4'
            )
        )

        fig.add_trace(
            go.Bar(
                x=all_windows,
                y=[best_results[best_results['Shipment Window'] == w]['Cost Savings'].values[0] if w in best_results[
                    'Shipment Window'].values else 0 for w in all_windows],
                name='Cost Savings',
                marker_color='#a9d6a9'
            )
        )

        # Add the line chart for Total Shipments on secondary y-axis
        fig.add_trace(
            go.Scatter(
                x=all_windows,
                y=[best_results[best_results['Shipment Window'] == w]['Total Shipments'].values[0] if w in best_results[
                    'Shipment Window'].values else None for w in all_windows],
                name='Total Shipments',
                mode='lines+markers',
                line=dict(color='#ff7f0e', width=2),
                marker=dict(size=8),
                hovertemplate='<b>Shipment Window</b>: %{x}<br>' +
                              '<b>Total Shipments</b>: %{y}<br>' +
                              '<b>Average Utilization</b>: %{text:.1f}%<extra></extra>',
                text=[best_results[best_results['Shipment Window'] == w]['Average Utilization'].values[0] if w in
                                                                                                             best_results[
                                                                                                                 'Shipment Window'].values else None
                      for w in all_windows],
            ),
            secondary_y=True
        )

        # Add text annotations for Percent Savings
        for w in all_windows:
            if w in best_results['Shipment Window'].values:
                row = best_results[best_results['Shipment Window'] == w].iloc[0]
                fig.add_annotation(
                    x=w,
                    y=row['Total Shipment Cost'] + row['Cost Savings'],
                    text=f"{row['Percent Savings']:.1f}%",
                    showarrow=False,
                    yanchor='bottom',
                    yshift=5,
                    font=dict(size=10)
                )

        # Update the layout
        fig.update_layout(
            barmode='stack',
            height=600,
            width=1050,
            # margin=dict(l=50, r=50, t=40, b=20),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )

        fig.update_xaxes(title_text='Shipment Window', tickmode='linear', dtick=1, range=[-0.5, 30.5])
        fig.update_yaxes(title_text='Cost (£)', secondary_y=False)
        fig.update_yaxes(title_text='Total Shipments', secondary_y=True)

        # Show the chart
        st.plotly_chart(fig, use_container_width=False)

    st.info("Analyzing the Results...")


def create_calendar_heatmap_before_vs_after(parameters):
    charts = create_heatmap_and_bar_charts_dynamic(parameters['all_consolidated_shipments'], parameters['filtered_df'],
                                           parameters['start_date'], parameters['end_date'])

    with st.expander("**HEATMAP ANALYSIS CHARTS(Before & After Consolidation)**"):
        years_in_range = set(pd.date_range(parameters['start_date'], parameters['end_date']).year)
        for year in [2023,2024,2025]:
            if year in years_in_range:
                chart_original, chart_consolidated, bar_comparison = charts[year]

                # Display heatmaps for the current year
                st.write(f"**Heatmaps for the year {year} (Before & After Consolidation):**")
                st.components.v1.html(chart_original.render_embed(), height=216, width=1000)
                st.components.v1.html(chart_consolidated.render_embed(), height=216, width=1000)

        # After the loop, you can add the interpretation section just once
        st.write("""
                    **Heatmap Interpretation:**

                    - **Dark Green Areas**: Indicate high shipment concentration on specific dates, showcasing peak activity where most orders are processed.
                    - **Lighter Green Areas**: Represent fewer or no shipments, highlighting potential inefficiencies in the initial shipment strategy before optimization.

                    **Before Consolidation:**

                    - Shipments were frequent but scattered across multiple days without strategic grouping.
                    - Increased costs due to multiple underutilized shipments.
                    - Truck utilization remained suboptimal, leading to excess operational expenses.

                    **After Consolidation:**

                    - Orders were intelligently grouped into fewer shipments, reducing the total number of trips while maintaining service levels.
                    - Optimized cost savings through better utilization and fewer underfilled shipments.
                    - Enhanced planning efficiency, enabling better decision-making for future shipment scheduling.
                    """)

def show_ui_cost_saving_agent(response_parameters):

    summary_text = (
        f"Optimizing outbound deliveries and identifying cost-saving opportunities involve analyzing various factors "
        f"such as order patterns, delivery routes, shipping costs, and consolidation opportunities.\n\n"

        f"On analyzing the data, I can provide some estimates of cost savings on the historical data if we were to "
        f"group orders to consolidate deliveries.\n\n"

        "**APPROACH TAKEN**\n\n"  # Ensure it's already in uppercase and same.
        f"To consolidate the deliveries, A heuristic approach was used, and the methodology is as follows:\n\n"

        f"**Group Shipments**: Orders are consolidated within a shipment window to reduce transportation costs while "
        f"maintaining timely deliveries. A shipment window represents the number of days prior to the current delivery "
        f"that the order could be potentially shipped, thus representing an opportunity to group it with earlier deliveries.\n\n"

        f"**Iterate Over Shipment Windows**: The model systematically evaluates all possible shipment windows, testing "
        f"different configurations to identify the most effective scheduling approach.\n\n"

        f"**Performance Metric Calculation**: Key performance metrics are assessed for each shipment window, including:\n"
        f"- **Cost savings**\n"
        f"- **Utilization rate**\n"
        f"- **CO2 emission reduction**\n\n"

        f"**Comparison and Selection**: After evaluating all configurations, the shipment window that maximizes cost savings "
        f"while maintaining operational efficiency is identified, and results are displayed as per the best parameter.\n\n"

        f"This method allows us to optimize logistics operations dynamically, ensuring that both financial and environmental "
        f"factors are balanced effectively."
    )

    with st.expander("**VIEW APPROACH OF COST CONSOLIDATION**", expanded=False):
        st.write(summary_text)


    create_shipment_window_vs_saving_plot(response_parameters["all_results"], response_parameters["best_params"])

    st.write(response_parameters["shipment_window_vs_saving_agent_msg"])
    st.info(f"Consolidating orders for window {int(response_parameters['window'])}..!")


    main_text = (
        f"Through extensive analysis, the OPTIMAL SHIPMENT WINDOW was determined to be **{response_parameters['best_params'][0]}**, "
        f"with a PALLET SIZE of **46** for **postcodes**: {response_parameters['selected_postcodes']} and **customers**: {response_parameters['selected_customers']}."
        f"These optimizations resulted in SIGNIFICANT EFFICIENCY IMPROVEMENTS:\n\n"

        f"**SHIPMENT WINDOW**: The most effective shipment window was identified as **{response_parameters['best_params'][0]} DAYS**.\n\n"

        f"**COST SAVINGS**: A reduction of **£{response_parameters['metrics']['Cost Savings']:,.1f}**, equating to an **£{response_parameters['metrics']['Percent Savings']:.1f}%** decrease in overall transportation costs.\n\n"

        f"**ORDER & SHIPMENT SUMMARY**:\n"
        f"- TOTAL ORDERS PROCESSED: **{response_parameters['metrics']['Total Orders']:,}** \n"
        f"- TOTAL SHIPMENTS MADE: **{response_parameters['metrics']['Total Shipments']:,}**\n\n"

        f"**UTILIZATION EFFICIENCY**:\n"
        f"- AVERAGE TRUCK UTILIZATION increased to **{response_parameters['metrics']['Average Utilization']:.1f}%**, ensuring fewer trucks operate at low capacity.\n\n"

        f"**ENVIRONMENTAL IMPACT**:\n"
        f"- CO2 EMISSIONS REDUCTION: A decrease of **{response_parameters['metrics']['CO2 Emission']:,.1f} Kg**, supporting sustainability efforts and reducing the carbon footprint.\n\n"

        f"These optimizations not only lead to substantial COST REDUCTIONS but also enhance OPERATIONAL SUSTAINABILITY, "
        f"allowing logistics operations to function more efficiently while MINIMIZING ENVIRONMENTAL IMPACT."
    )
    with st.expander("**IDENTIFIED COST SAVINGS AND KEY PERFORMANCE INDICATORS(KPIs)**", expanded=False):
        st.write(main_text)

    comparison_df_dict = response_parameters['comparison_df'].to_dict()

    # Create three columns for before, after, and change metrics
    st.info("Comparing before and after consolidation...")
    col1, col2, col3 = st.columns(3)

    # Style for metric display
    metric_style = """
                        <div style="
                            background-color: #f0f2f6;
                            padding: 0px;
                            border-radius: 5px;
                            margin: 5px 0;
                        ">
                            <span style="font-weight: bold;">{label}:</span> {value}
                        </div>
                    """

    # Style for percentage changes
    change_style = """
                        <div style="
                            background-color: #e8f0fe;
                            padding: 0px;
                            border-radius: 5px;
                            margin: 5px 0;
                            display: flex;
                            justify-content: space-between;
                            align-items: center;
                        ">
                            <span style="font-weight: bold;">{label}:</span>
                            <span style="color: {color}; font-weight: bold;">{value:+.1f}%</span>
                        </div>
                    """

    # Before consolidation metrics

    with col1:
        st.markdown("##### Before Consolidation")
        st.markdown(metric_style.format(
            label="Days Shipped",
            value=f"{comparison_df_dict['Before']['Days']:,}"
        ), unsafe_allow_html=True)
        st.markdown(metric_style.format(
            label="Pallets Shipped per Day",
            value=f"{comparison_df_dict['Before']['Pallets Per Day']:.1f}"
        ), unsafe_allow_html=True)
        st.markdown(metric_style.format(
            label="Pallets per Shipment",
            value=f"{comparison_df_dict['Before']['Pallets per Shipment']:.1f}"
        ), unsafe_allow_html=True)

    # After consolidation metrics
    with col2:
        st.markdown("##### After Consolidation")
        st.markdown(metric_style.format(
            label="Days Shipped",
            value=f"{comparison_df_dict['After']['Days']:,}"
        ), unsafe_allow_html=True)
        st.markdown(metric_style.format(
            label="Pallets Shipped per Day",
            value=f"{comparison_df_dict['After']['Pallets Per Day']:.1f}"
        ), unsafe_allow_html=True)
        st.markdown(metric_style.format(
            label="Pallets per Shipment",
            value=f"{comparison_df_dict['After']['Pallets per Shipment']:.1f}"
        ), unsafe_allow_html=True)

    # Percentage changes
    with col3:
        st.markdown("##### Percentage Change")
        st.markdown(change_style.format(
            label="Days Shipped",
            value=comparison_df_dict['% Change']['Days'],
            color="blue" if comparison_df_dict['% Change']['Days'] > 0 else "green"
        ), unsafe_allow_html=True)
        st.markdown(change_style.format(
            label="Pallets Shipped per Day",
            value=comparison_df_dict['% Change']['Pallets Per Day'],
            color="green" if comparison_df_dict['% Change']['Pallets Per Day'] > 0 else "red"
        ), unsafe_allow_html=True)
        st.markdown(change_style.format(
            label="Pallets per Shipment",
            value=comparison_df_dict['% Change']['Pallets per Shipment'],
            color="green" if comparison_df_dict['% Change']['Pallets per Shipment'] > 0 else "red"
        ), unsafe_allow_html=True)

    create_calendar_heatmap_before_vs_after(response_parameters)


def show_ui_cost_saving_agent_static(response_parameters):
    best_scenario = response_parameters["best_scenario"]
    total_shipment_cost = response_parameters["total_shipment_cost"]
    aggregated_data = response_parameters["aggregated_data"]
    all_results = response_parameters["all_results"]
    consolidated_data = response_parameters["consolidated_data"]

    # Display results using Streamlit
    metric_style = """
                    <div style="
                        background-color: #f0f2f6;
                        padding: 10px;
                        border-radius: 5px;
                        margin: 5px 0;
                    ">
                        <span style="font-weight: bold;">{label}:</span> {value}
                    </div>
                """

    change_style = """
                    <div style="
                        background-color: #e8f0fe;
                        padding: 10px;
                        border-radius: 5px;
                        margin: 5px 0;
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                    ">
                        <span style="font-weight: bold;">{label}:</span>
                        <span style="color: {color}; font-weight: bold;">{value:+.1f}%</span>
                    </div>
                """

    # Create columns
    st.write(f"**For the Best cost saving Scenario : {best_scenario['scenario']} ⬇️**")
    col1, col2, col3 = st.columns(3)

    # Before Consolidation
    with col1:
        st.markdown("##### Before Consolidation")
        st.markdown(metric_style.format(
            label="Total Shipment Cost",
            value=f"€{total_shipment_cost:,.2f}"
        ), unsafe_allow_html=True)
        st.markdown(metric_style.format(
            label="No of Shipments",
            value=f"{len(aggregated_data.index):,}"
        ), unsafe_allow_html=True)
        st.markdown(metric_style.format(
            label="Avg Pallets",
            value=f"{round(aggregated_data['Total Pallets'].mean(), 2)}"
        ), unsafe_allow_html=True)

    # After Consolidation
    with col2:
        st.markdown(f"##### After Consolidation")
        st.markdown(metric_style.format(
            label="Total Consolidated Shipment Cost",
            value=f"€{best_scenario['total_consolidated_cost']:,.2f}"
        ), unsafe_allow_html=True)
        st.markdown(metric_style.format(
            label="No of Shipments",
            value=f"{best_scenario['num_shipments']:,}"
        ), unsafe_allow_html=True)
        st.markdown(metric_style.format(
            label="Avg Pallets",
            value=f"{best_scenario['avg_pallets']}"
        ), unsafe_allow_html=True)

    # Percentage Changes
    with col3:
        st.markdown("##### Percentage Change")

        # Calculate percentage changes
        shipment_cost_change = ((best_scenario[
                                     'total_consolidated_cost'] - total_shipment_cost) / total_shipment_cost) * 100
        num_shipments_change = ((best_scenario['num_shipments'] - len(aggregated_data.index)) / len(
            aggregated_data.index)) * 100
        avg_pallets_change = ((best_scenario['avg_pallets'] - aggregated_data['Total Pallets'].mean()) /
                              aggregated_data['Total Pallets'].mean()) * 100

        # Display percentage changes
        st.markdown(change_style.format(
            label="Shipment Cost",
            value=shipment_cost_change,
            color="green" if shipment_cost_change < 0 else "red"  # Negative change is good (cost savings)
        ), unsafe_allow_html=True)
        st.markdown(change_style.format(
            label="No of Shipments",
            value=num_shipments_change,
            color="green" if num_shipments_change < 0 else "red"  # Negative change is good (fewer shipments)
        ), unsafe_allow_html=True)
        st.markdown(change_style.format(
            label="Avg Pallets",
            value=avg_pallets_change,
            color="green" if avg_pallets_change > 0 else "red"
            # Positive change is good (more pallets per shipment)
        ), unsafe_allow_html=True)

    st.write(" ")
    # Step 7: Display remaining scenarios
    with st.expander("Remaining delivery scenarios ⬇️"):
        sorted_results = all_results.reset_index(drop=True)
        st.dataframe(sorted_results, use_container_width=True)

    charts = create_heatmap_and_bar_charts_static(consolidated_data, aggregated_data, response_parameters['start_date'],
                                                  response_parameters['end_date'])
    st.write(" ")
    with st.expander("Heatmap Analysis Charts(Before & After Consolidation)"):
        for year in [2023, 2024,2025]:
            years_in_range = set(pd.date_range(response_parameters['start_date'], response_parameters['end_date']).year)
            if year in years_in_range:
                chart_original, chart_consolidated, bar_comparison = charts[year]

                # Display heatmaps for the current year
                st.write(f"**Visualisation using Heatmaps (Before & After Consolidation):**")
                st.components.v1.html(chart_original.render_embed(), height=216, width=1000)
                st.components.v1.html(chart_consolidated.render_embed(), height=216, width=1000)


