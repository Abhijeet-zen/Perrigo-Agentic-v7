"""
__init__.py for src/core

This package provides core utilities for cost optimization and static cost optimization.

It includes:
- Functions for optimizing shipment costs dynamically.
- Static cost optimization functions such as consolidations and visualization tools.
"""

from src.core.order_consolidation.dynamic_consolidation import (
    get_chatgpt_response,
    ask_openai,
    get_parameters_values,
    get_filtered_data,
    calculate_metrics,
    analyze_consolidation_distribution,
    create_utilization_chart
)

from src.core.order_consolidation.static_consolidation import (
    cost_of_columns,
    consolidations_day_mapping,
    consolidate_shipments,
    create_consolidated_shipments_calendar_static,
    create_original_orders_calendar_static,
    create_heatmap_and_bar_charts_static
)

from src.core.order_consolidation.consolidation_ui import (
    timed_call
)

# Load prompt templates from the prompt_templates package
try:
    from prompt_templates import load_template  # Ensure this import works
    COST_PARAMETERS_PROMPT = load_template("cost_parameters_prompt.txt")
except Exception as e:
    COST_PARAMETERS_PROMPT = "Cost parameters prompt not found. Error: " + str(e)

try:
    CUSTOMER_POSTCODE_MATCHING_PROMPT = load_template("customer_postcode_matching_prompt.txt")
except Exception as e:
    CUSTOMER_POSTCODE_MATCHING_PROMPT = "Customer postcode matching prompt not found. Error: " + str(e)

# Expose all functions and variables
__all__ = [
    # From core.py
    "get_chatgpt_response",
    "ask_openai",
    "get_parameters_values",
    "get_filtered_data",
    "calculate_metrics",
    "analyze_consolidation_distribution",
    "create_utilization_chart",

    # From static.py
    "cost_of_columns",
    "consolidations_day_mapping",
    "consolidate_shipments",
    "create_consolidated_shipments_calendar_static",
    "create_original_orders_calendar_static",
    "create_heatmap_and_bar_charts_static",

    # Prompt templates
    "COST_PARAMETERS_PROMPT",
    "CUSTOMER_POSTCODE_MATCHING_PROMPT"

    "timed_call"
]

