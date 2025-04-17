import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

import calendar
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pyecharts import options as opts
from pyecharts.charts import Calendar, Page
from pyecharts.globals import ThemeType
from pyecharts.commons.utils import JsCode

consolidations_day_mapping = {
    # 5-day scenarios
    'Mon_Tue_Wed_Thu_Fri': {
        'Mon': 0,
        'Tue': 0,
        'Wed': 0,
        'Thu': 0,
        'Fri': 0,
        'Sat': -1,
        'Sun': -2
    },
    'Mon_Tue_Wed_Thu_Sat': {
        'Mon': 0,
        'Tue': 0,
        'Wed': 0,
        'Thu': 0,
        'Fri': -1,
        'Sat': 0,
        'Sun': -1
    },
    'Mon_Tue_Wed_Thu_Sun': {
        'Mon': 0,
        'Tue': 0,
        'Wed': 0,
        'Thu': 0,
        'Fri': -1,
        'Sat': -2,
        'Sun': 0
    },
    'Mon_Tue_Wed_Fri_Sat': {
        'Mon': 0,
        'Tue': 0,
        'Wed': 0,
        'Thu': -1,
        'Fri': 0,
        'Sat': 0,
        'Sun': -1
    },
    'Mon_Tue_Wed_Fri_Sun': {
        'Mon': 0,
        'Tue': 0,
        'Wed': 0,
        'Thu': -1,
        'Fri': 0,
        'Sat': -1,
        'Sun': 0
    },
    'Mon_Tue_Thu_Fri_Sat': {
        'Mon': 0,
        'Tue': 0,
        'Wed': -1,
        'Thu': 0,
        'Fri': 0,
        'Sat': 0,
        'Sun': -1
    },
    'Mon_Tue_Thu_Fri_Sun': {
        'Mon': 0,
        'Tue': 0,
        'Wed': -1,
        'Thu': 0,
        'Fri': 0,
        'Sat': -1,
        'Sun': 0
    },
    'Mon_Wed_Thu_Fri_Sat': {
        'Mon': 0,
        'Tue': -1,
        'Wed': 0,
        'Thu': 0,
        'Fri': 0,
        'Sat': 0,
        'Sun': -1
    },
    'Mon_Wed_Thu_Fri_Sun': {
        'Mon': 0,
        'Tue': -1,
        'Wed': 0,
        'Thu': 0,
        'Fri': 0,
        'Sat': -1,
        'Sun': 0
    },
    'Tue_Wed_Thu_Fri_Sat': {
        'Mon': -2,
        'Tue': 0,
        'Wed': 0,
        'Thu': 0,
        'Fri': 0,
        'Sat': 0,
        'Sun': -1
    },
    'Tue_Wed_Thu_Fri_Sun': {
        'Mon': -1,
        'Tue': 0,
        'Wed': 0,
        'Thu': 0,
        'Fri': 0,
        'Sat': -1,
        'Sun': 0
    },

    # 4-day scenarios
    'Mon_Tue_Wed_Thu': {
        'Mon': 0,
        'Tue': 0,
        'Wed': 0,
        'Thu': 0,
        'Fri': -1,
        'Sat': -2,
        'Sun': -3
    },
    'Mon_Tue_Wed_Fri': {
        'Mon': 0,
        'Tue': 0,
        'Wed': 0,
        'Thu': -1,
        'Fri': 0,
        'Sat': -1,
        'Sun': -2
    },
    'Mon_Tue_Wed_Sat': {
        'Mon': 0,
        'Tue': 0,
        'Wed': 0,
        'Thu': -1,
        'Fri': -2,
        'Sat': 0,
        'Sun': -1
    },
    'Mon_Tue_Wed_Sun': {
        'Mon': 0,
        'Tue': 0,
        'Wed': 0,
        'Thu': -1,
        'Fri': -2,
        'Sat': -3,
        'Sun': 0
    },
    'Mon_Tue_Thu_Fri': {
        'Mon': 0,
        'Tue': 0,
        'Wed': -1,
        'Thu': 0,
        'Fri': 0,
        'Sat': -1,
        'Sun': -2
    },
    'Mon_Tue_Thu_Sat': {
        'Mon': 0,
        'Tue': 0,
        'Wed': -1,
        'Thu': 0,
        'Fri': -1,
        'Sat': 0,
        'Sun': -1
    },
    'Mon_Tue_Thu_Sun': {
        'Mon': 0,
        'Tue': 0,
        'Wed': -1,
        'Thu': 0,
        'Fri': -1,
        'Sat': -2,
        'Sun': 0
    },
    'Mon_Wed_Thu_Fri': {
        'Mon': 0,
        'Tue': -1,
        'Wed': 0,
        'Thu': 0,
        'Fri': 0,
        'Sat': -1,
        'Sun': -2
    },
    'Mon_Wed_Thu_Sat': {
        'Mon': 0,
        'Tue': -1,
        'Wed': 0,
        'Thu': 0,
        'Fri': -1,
        'Sat': 0,
        'Sun': -1
    },
    'Mon_Wed_Thu_Sun': {
        'Mon': 0,
        'Tue': -1,
        'Wed': 0,
        'Thu': 0,
        'Fri': -1,
        'Sat': -2,
        'Sun': 0
    },
    'Tue_Wed_Thu_Fri': {
        'Mon': -3,
        'Tue': 0,
        'Wed': 0,
        'Thu': 0,
        'Fri': 0,
        'Sat': -1,
        'Sun': -2
    },
    'Tue_Wed_Thu_Sat': {
        'Mon': -2,
        'Tue': 0,
        'Wed': 0,
        'Thu': 0,
        'Fri': -1,
        'Sat': 0,
        'Sun': -1
    },
    'Tue_Wed_Thu_Sun': {
        'Mon': -1,
        'Tue': 0,
        'Wed': 0,
        'Thu': 0,
        'Fri': -1,
        'Sat': -2,
        'Sun': 0
    },

    # 3-day scenarios
    'Mon_Tue_Wed': {
        'Mon': 0,
        'Tue': 0,
        'Wed': 0,
        'Thu': -1,
        'Fri': -2,
        'Sat': -3,
        'Sun': -4
    },
    'Mon_Tue_Thu': {
        'Mon': 0,
        'Tue': 0,
        'Wed': -1,
        'Thu': 0,
        'Fri': -1,
        'Sat': -2,
        'Sun': -3
    },
    'Mon_Tue_Fri': {
        'Mon': 0,
        'Tue': 0,
        'Wed': -1,
        'Thu': -2,
        'Fri': 0,
        'Sat': -1,
        'Sun': -2
    },
    'Mon_Tue_Sat': {
        'Mon': 0,
        'Tue': 0,
        'Wed': -1,
        'Thu': -2,
        'Fri': -3,
        'Sat': 0,
        'Sun': -1
    },
    'Mon_Tue_Sun': {
        'Mon': 0,
        'Tue': 0,
        'Wed': -1,
        'Thu': -2,
        'Fri': -3,
        'Sat': -4,
        'Sun': 0
    },
    'Mon_Wed_Thu': {
        'Mon': 0,
        'Tue': -1,
        'Wed': 0,
        'Thu': 0,
        'Fri': -1,
        'Sat': -2,
        'Sun': -3
    },
    'Mon_Wed_Fri': {
        'Mon': 0,
        'Tue': -1,
        'Wed': 0,
        'Thu': -1,
        'Fri': 0,
        'Sat': -1,
        'Sun': -2
    },
    'Mon_Wed_Sat': {
        'Mon': 0,
        'Tue': -1,
        'Wed': 0,
        'Thu': -1,
        'Fri': -2,
        'Sat': 0,
        'Sun': -1
    },
    'Mon_Wed_Sun': {
        'Mon': 0,
        'Tue': -1,
        'Wed': 0,
        'Thu': -1,
        'Fri': -2,
        'Sat': -3,
        'Sun': 0
    },
    'Tue_Wed_Thu': {
        'Mon': -4,
        'Tue': 0,
        'Wed': 0,
        'Thu': 0,
        'Fri': -1,
        'Sat': -2,
        'Sun': -3
    },
    'Tue_Wed_Fri': {
        'Mon': -3,
        'Tue': 0,
        'Wed': 0,
        'Thu': -1,
        'Fri': 0,
        'Sat': -1,
        'Sun': -2
    },
    'Tue_Wed_Sat': {
        'Mon': -2,
        'Tue': 0,
        'Wed': 0,
        'Thu': -1,
        'Fri': -2,
        'Sat': 0,
        'Sun': -1
    },
    'Tue_Wed_Sun': {
        'Mon': -1,
        'Tue': 0,
        'Wed': 0,
        'Thu': -1,
        'Fri': -2,
        'Sat': -3,
        'Sun': 0
    },

    # 2-day scenarios
    'Mon_Tue': {
        'Mon': 0,
        'Tue': 0,
        'Wed': -1,
        'Thu': -2,
        'Fri': -3,
        'Sat': -4,
        'Sun': -5
    },
    'Mon_Wed': {
        'Mon': 0,
        'Tue': -1,
        'Wed': 0,
        'Thu': -1,
        'Fri': -2,
        'Sat': -3,
        'Sun': -4
    },
    'Mon_Thu': {
        'Mon': 0,
        'Tue': -1,
        'Wed': -2,
        'Thu': 0,
        'Fri': -1,
        'Sat': -2,
        'Sun': -3
    },
    'Mon_Fri': {
        'Mon': 0,
        'Tue': -1,
        'Wed': -2,
        'Thu': -3,
        'Fri': 0,
        'Sat': -1,
        'Sun': -2
    },
    'Mon_Sat': {
        'Mon': 0,
        'Tue': -1,
        'Wed': -2,
        'Thu': -3,
        'Fri': -4,
        'Sat': 0,
        'Sun': -1
    },
    'Mon_Sun': {
        'Mon': 0,
        'Tue': -1,
        'Wed': -2,
        'Thu': -3,
        'Fri': -4,
        'Sat': -5,
        'Sun': 0
    },
    'Tue_Wed': {
        'Mon': -5,
        'Tue': 0,
        'Wed': 0,
        'Thu': -1,
        'Fri': -2,
        'Sat': -3,
        'Sun': -4
    },
    'Tue_Thu': {
        'Mon': -4,
        'Tue': 0,
        'Wed': -1,
        'Thu': 0,
        'Fri': -1,
        'Sat': -2,
        'Sun': -3
    },
    'Tue_Fri': {
        'Mon': -3,
        'Tue': 0,
        'Wed': -1,
        'Thu': -2,
        'Fri': 0,
        'Sat': -1,
        'Sun': -2
    },
    'Tue_Sat': {
        'Mon': -2,
        'Tue': 0,
        'Wed': -1,
        'Thu': -2,
        'Fri': -3,
        'Sat': 0,
        'Sun': -1
    },
    'Tue_Sun': {
        'Mon': -1,
        'Tue': 0,
        'Wed': -1,
        'Thu': -2,
        'Fri': -3,
        'Sat': -4,
        'Sun': 0
    },
    'Wed_Thu': {
        'Mon': -4,
        'Tue': -5,
        'Wed': 0,
        'Thu': 0,
        'Fri': -1,
        'Sat': -2,
        'Sun': -3
    },
    'Wed_Fri': {
        'Mon': -3,
        'Tue': -4,
        'Wed': 0,
        'Thu': -1,
        'Fri': 0,
        'Sat': -1,
        'Sun': -2
    },
    'Wed_Sat': {
        'Mon': -2,
        'Tue': -3,
        'Wed': 0,
        'Thu': -1,
        'Fri': -2,
        'Sat': 0,
        'Sun': -1
    },
    'Wed_Sun': {
        'Mon': -1,
        'Tue': -2,
        'Wed': 0,
        'Thu': -1,
        'Fri': -2,
        'Sat': -3,
        'Sun': 0
    },

    # 1-day scenarios
    'Only_Mon': {
        'Mon': 0,
        'Tue': -1,
        'Wed': -2,
        'Thu': -3,
        'Fri': -4,
        'Sat': -5,
        'Sun': -6
    },
    'Only_Tue': {
        'Mon': -6,
        'Tue': 0,
        'Wed': -1,
        'Thu': -2,
        'Fri': -3,
        'Sat': -4,
        'Sun': -5
    },
    'Only_Wed': {
        'Mon': -5,
        'Tue': -6,
        'Wed': 0,
        'Thu': -1,
        'Fri': -2,
        'Sat': -3,
        'Sun': -4
    },
    'Only_Thu': {
        'Mon': -4,
        'Tue': -5,
        'Wed': -6,
        'Thu': 0,
        'Fri': -1,
        'Sat': -2,
        'Sun': -3
    },
    'Only_Fri': {
        'Mon': -3,
        'Tue': -4,
        'Wed': -5,
        'Thu': -6,
        'Fri': 0,
        'Sat': -1,
        'Sun': -2
    },
    'Only_Sat': {
        'Mon': -2,
        'Tue': -3,
        'Wed': -4,
        'Thu': -5,
        'Fri': -6,
        'Sat': 0,
        'Sun': -1

    },
    'Only_Sun': {
        'Mon': -1,
        'Tue': -2,
        'Wed': -3,
        'Thu': -4,
        'Fri': -5,
        'Sat': -6,
        'Sun': 0

    }
}


def create_consolidated_shipments_calendar_static(consolidated_df):
    # Group by UPDATED_DATE and calculate consolidated_shipment_cost and Total Pallets
    df_consolidated = consolidated_df.groupby('UPDATED_DATE').agg({
        'consolidated_shipment_cost': 'sum',  # Sum of shipment costs
        'Total Pallets': 'sum'  # Sum of total pallets
    }).reset_index()
    df_consolidated.columns = ['UPDATED_DATE', 'consolidated_shipment_cost', 'Total Pallets']

    # Split data by year
    df_2023 = df_consolidated[df_consolidated['UPDATED_DATE'].dt.year == 2023]
    df_2024 = df_consolidated[df_consolidated['UPDATED_DATE'].dt.year == 2024]
    df_2025 = df_consolidated[df_consolidated['UPDATED_DATE'].dt.year == 2025]

    calendar_data_2023 = df_2023[['UPDATED_DATE', 'consolidated_shipment_cost', 'Total Pallets']].values.tolist()
    calendar_data_2024 = df_2024[['UPDATED_DATE', 'consolidated_shipment_cost', 'Total Pallets']].values.tolist()
    calendar_data_2025 = df_2025[['UPDATED_DATE', 'consolidated_shipment_cost', 'Total Pallets']].values.tolist()

    print("calendar data",calendar_data_2024,type(calendar_data_2024))
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
                    title=f"Calendar Heatmap for Consolidated Shipments ({year})"),
                visualmap_opts=opts.VisualMapOpts(
                    max_=max(item[2] for item in data) if data else 0,  # Use Total Pallets for heatmap
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
                                   '<br/>Consolidated Shipment Cost: ' + p.data[1] +
                                   '<br/>Total Pallets: ' + p.data[2];
                        }
                        """
                    )
                )
            )
        )

    calendar_2023 = create_calendar(calendar_data_2023, 2023)
    calendar_2024 = create_calendar(calendar_data_2024, 2024)
    calendar_2025 = create_calendar(calendar_data_2025, 2025)

    return calendar_2023, calendar_2024 , calendar_2025


def create_original_orders_calendar_static(original_df):
    # Group by SHIPPED_DATE and calculate the number of orders and total pallets
    df_original = original_df.groupby('SHIPPED_DATE').agg({
        'ORDER_ID': 'count',  # Count of orders
        'Total Pallets': 'sum'  # Sum of total pallets
    }).reset_index()
    df_original.columns = ['SHIPPED_DATE', 'Orders Shipped', 'Total Pallets']

    # Split data by year
    df_2023 = df_original[df_original['SHIPPED_DATE'].dt.year == 2023]
    df_2024 = df_original[df_original['SHIPPED_DATE'].dt.year == 2024]
    df_2025 = df_original[df_original['SHIPPED_DATE'].dt.year == 2025]

    calendar_data_2023 = df_2023[['SHIPPED_DATE', 'Orders Shipped', 'Total Pallets']].values.tolist()
    calendar_data_2024 = df_2024[['SHIPPED_DATE', 'Orders Shipped', 'Total Pallets']].values.tolist()
    calendar_data_2025 = df_2025[['SHIPPED_DATE', 'Orders Shipped', 'Total Pallets']].values.tolist()

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
                title_opts=opts.TitleOpts(title=f"Calendar Heatmap for Original Orders ({year})"),
                visualmap_opts=opts.VisualMapOpts(
                    max_=max(item[1] for item in data) if data else 0,  # Use Orders Shipped for heatmap
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
                            return 'Date: ' + day + '/' + month + '/' + year + 
                                   '<br/>Orders Shipped: ' + p.data[1] +
                                   '<br/>Total Pallets: ' + p.data[2];
                        }
                        """
                    )
                )
            )
        )

    calendar_2023 = create_calendar(calendar_data_2023, 2023)
    calendar_2024 = create_calendar(calendar_data_2024, 2024)
    calendar_2025 = create_calendar(calendar_data_2025, 2025)

    return calendar_2023, calendar_2024 , calendar_2025


def create_heatmap_and_bar_charts_static(consolidated_df, original_df, start_date, end_date):
    # Create calendar charts (existing code)
    chart_original_2023, chart_original_2024 , chart_original_2025 = create_original_orders_calendar_static(original_df)
    chart_consolidated_2023, chart_consolidated_2024  , chart_consolidated_2025= create_consolidated_shipments_calendar_static(consolidated_df)

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

def calculate_cost(total_pallets, prod_type, postcode, rate_card):
    """
    Calculate the cost based on the number of pallets, product type, and postcode.
    """

    # Filter the rate card based on product type and postcode
    filtered_rate = rate_card[(rate_card['PROD TYPE'] == prod_type) &
                              (rate_card['SHORT_POSTCODE'] == postcode)]

    if filtered_rate.empty:
        return None

    # Initialize total cost
    total_cost = 0

    # Handle cost calculation
    if total_pallets <= 52:
        cost_column = f"COST PER {int(total_pallets)} PALLET{'S' if total_pallets > 1 else ''}"
        if cost_column in filtered_rate.columns:
            total_cost = filtered_rate[cost_column].values[0]
    else:
        # Split into full batches of 52 pallets and remaining pallets
        full_batches = total_pallets // 52
        remaining_pallets = total_pallets % 52

        # Calculate cost for full batches
        batch_column = "COST PER 52 PALLETS"
        if batch_column in filtered_rate.columns:
            total_cost = filtered_rate[batch_column].values[0]

        # Calculate cost for remaining pallets
        if remaining_pallets > 0:
            remaining_column = f"COST PER {int(remaining_pallets)} PALLET{'S' if remaining_pallets > 1 else ''}"
            if remaining_column in filtered_rate.columns:
                total_cost = (total_cost+filtered_rate[remaining_column].values[0])/2

    # Return the total cost
    return total_cost if total_cost > 0 else None


def cost_of_columns(filter_data, rate_card):

    aggregated_data = filter_data.groupby(
        ['PROD TYPE', 'SHORT_POSTCODE', 'ORDER_ID', 'SHIPPED_DATE'], as_index=False
    ).agg({'Total Pallets': 'sum', 'Distance': 'first', 'NAME': 'first'})


    aggregated_data['shipment_cost'] = aggregated_data.apply(
        lambda row: calculate_cost(row['Total Pallets'], row['PROD TYPE'], row['SHORT_POSTCODE'], rate_card) * (row['Total Pallets']),
        axis=1
    )
    return aggregated_data , aggregated_data['shipment_cost'].sum()


def get_updated_delivery_date(current_date, day_mapping):
    current_day = current_date.strftime('%a')
    updated_date = current_date + timedelta(day_mapping.get(current_day, 0))
    return updated_date


def consolidate_shipments(aggregated_data, rate_card, day_mapping):

    aggregated_data['UPDATED_DATE'] = aggregated_data['SHIPPED_DATE'].apply(
        lambda x: get_updated_delivery_date(x, day_mapping)
    )

    consolidated_data = aggregated_data.groupby(
        ['PROD TYPE', 'SHORT_POSTCODE', 'UPDATED_DATE'], as_index=False
    ).agg({'Total Pallets': 'sum', 'Distance': 'first', 'NAME': 'first'})

    # Calculate the consolidated shipment cost
    consolidated_data['consolidated_shipment_cost'] = consolidated_data.apply(
        lambda row: calculate_cost(row['Total Pallets'], row['PROD TYPE'], row['SHORT_POSTCODE'], rate_card)  * (row['Total Pallets']),
        axis=1
    )
    return consolidated_data ,consolidated_data['consolidated_shipment_cost'].sum()