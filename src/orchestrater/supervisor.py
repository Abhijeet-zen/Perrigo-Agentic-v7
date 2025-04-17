"""
supervisor.py

This module defines the Multi-Agent Supervisor, responsible for managing 
the conversation flow between multiple agents and routing user queries.
"""
import re
from dotenv import load_dotenv, find_dotenv
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.utils.openai_api import get_supervisor_llm
from src.utils.load_templates import load_template

# Load environment variables
_ = load_dotenv(find_dotenv())

supervisor_prompt = load_template("supervisor_prompt.txt")

llm = get_supervisor_llm()

# Define the members
members = [
    {
        "agent_name": "Insights Agent", 
        "description": 
        """Insights Agent is responsible for analyzing shipment data to generate insights. 
         It handles tasks such as performing exploratory data analysis (EDA), calculating summary statistics, 
         identifying trends, comparing metrics across different dimensions (e.g., users, regions), and generating 
         visualizations to help understand shipment-related patterns and performance."""},
    {
        "agent_name": "Dynamic Cost Optimization Agent",
        "description": 
        """ The Dynamic Cost Optimization Agent is responsible for analyzing shipment cost-related data and recommending 
        strategies to reduce or optimize costs. This agent handles tasks such as identifying cost-saving 
        opportunities, calculating the optimal number of trips, performing scenario-based cost optimizations 
        (e.g., varying consolidation windows, truck capacity adjustments), and providing benchmarks and 
        comparisons between current and optimized operations. The agent also calculates key performance 
        metrics like cost per pallet, truck utilization rate, and cost savings over time. This agent is 
        called when the user asks about shipment cost reduction or optimization scenarios."""},
    {
        "agent_name": "Static Cost Optimization Agent",
        "description":
        """The Static Cost Optimization Agent is designed to analyze and optimize shipment costs by 
        evaluating (number of days of delivery) scenarios before and after consolidation. Using a Rate Card (which includes product type, short postcode, and cost per pallet),
        the agent calculates the base shipment costs. To maximize cost savings, the agent evaluates multiple delivery
        day scenarios (e.g., 5-day, 4-day, or 3-day delivery options).By applying consolidation day mappings, the agent
        aggregates shipments into fewer deliveries, reducing overall costs. The results include: Total shipment costs before and after consolidation ,
        Percentage savings achieved ,Key metrics such as the number of shipments and average pallets per shipment.
        This tool empowers users to identify the most cost-effective delivery strategies while maintaining operational efficiency.""" },
    {
        "agent_name": "Generate Scenario Agent", 
        "description": 
        """Generate Scenario Agent is responsible for creating and analyzing "what-if" scenarios based on 
        user-defined parameters. This agent helps compare the outcomes of various decisions or actions, such 
        as the impact of increasing truck capacity, changing shipment consolidation strategies, or exploring 
        different operational scenarios. It can model changes in the system and assess the consequences of 
        those changes to support decision-making and optimization. This agent is called when the user asks 
        about scenario generation, comparisons of different outcomes, or analysis of hypothetical situations."""},
    {
        "agent_name": "SELF_RESPONSE",
        "description": 
        """Use this option when you can directly answer the user's question without specialized data analysis.
        This is appropriate for:
        1. General questions about the system's capabilities
        2. Clarification questions
        3. Simple information requests that don't require data analysis
        4. Explanations of concepts or terms
        5. Help with formulating questions for specialized agents
        When selecting this option, you should provide a complete, helpful response to the user's query."""}
]

# ---------------------- Define Routing Logic ----------------------

# Define available options for the supervisor to choose from
options = ["FINISH"] + [mem["agent_name"] for mem in members]

# Generate structured agent information
members_info = "\n".join([f"{member['agent_name']}: {member['description']}" for member in members])

# Format the full prompt with agent details
final_prompt = supervisor_prompt + "\nHere is the information about the different agents available:\n" + members_info
final_prompt += """
Think step-by-step before choosing the next agent or deciding to answer directly. 

Examples of when to use SELF_RESPONSE:
- "Can you explain what the Insights Agentdoes?"
- "What kind of data does this system analyze?"
- "I'm not sure how to phrase my question about cost optimization"
- "What's the difference between Static and Dynamic Cost Optimization?"

Examples of when to route to specialized agents:
- "Analyze the shipment data and tell me which postcode has the highest delivery cost" (Insights Agent)
- "How can we optimize our delivery schedule to reduce costs?" (Cost Optimization Agents)
- "What if we increased our truck capacity by 20%?" (Generate Scenario Agent)

If needed, reflect on responses and adjust your approach and finally provide response.
"""


# Define LangChain prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", final_prompt.strip()),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Define the routing function schema
function_def = {
    "name": "route",
    "description": "Select the next role based on reasoning.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "thought_process": {
                "title": "Thought Process and Response",
                "type": "string",
                "description": "Step-by-step reasoning behind the decision and reply to the question."
            },
            "next": {
                "title": "Next",
                "anyOf": [{"enum": options}],
                "description": "The next agent to call or SELF_RESPONSE if answering directly."
            },
            "direct_response": {
                "title": "Direct Response",
                "type": "string",
                "description": "The direct response to provide to the user when SELF_RESPONSE is selected."
            }
        },
        "required": ["thought_process", "next","direct_response"],
    },
}

# ---------------------- Create Supervisor Chain ----------------------

supervisor_chain = (
    prompt
    | llm.bind_functions(functions=[function_def], function_call="route")
    | JsonOutputFunctionsParser()
)
