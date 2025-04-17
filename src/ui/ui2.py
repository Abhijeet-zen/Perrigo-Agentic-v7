"""
ui2.py

Streamlit-based UI for the multi-agent generative AI system.
"""
import os
import time

import markdown
from markupsafe import Markup

import uuid
import pandas as pd
import streamlit as st
import tiktoken
import openai
from langchain_core.messages import HumanMessage
from langgraph.graph.message import add_messages

from src.orchestrater.MultiAgentGraph import multi_agent_graph

# ---------------------- Helper Functions ----------------------

def display_saved_plot(plot_path: str):
    """
    Loads and displays a saved plot from the given path in a Streamlit app.

    Args:
        plot_path (str): Path to the saved plot image.
    """
    if os.path.exists(plot_path):
        st.image(plot_path, caption="Generated Plot", use_column_width=True)
    else:
        st.error(f"Plot not found at {plot_path}")

def reset_app_state():
    """Reset the app state when the data source changes."""
    st.session_state.initialized = False
    st.session_state.pop('df', None)

def load_data_file(filename):
    """Load a CSV file with automatic date parsing."""
    try:
        date_columns = [col for col in pd.read_csv(filename, nrows=1).columns if 'date' in col.lower()]
        return pd.read_csv(filename, parse_dates=date_columns, dayfirst=True)
    except Exception as e:
        st.error(f"Error loading {filename}: {e}")
        return None

# ---------------------- Sidebar Setup ----------------------

def setup_sidebar():
    """Set up sidebar with API key input and data source selection."""
    with st.sidebar.expander("📂 Select Data Source", expanded=False):
        api_key = st.session_state.get("OPENAI_API_KEY", "")

        Outbound_data = os.path.join("src", "data", "Outbound_Data.csv")
        Inventory_Data = os.path.join("src", "data", "Inventory_Batch.csv")
        Inbound_Data = os.path.join("src", "data", "Inbound_Data.csv")  # Fixing incorrect file path

        data_files = {
            'Outbound_Data.csv': Outbound_data,
            'Inventory_Batch.csv': Inventory_Data,
            'Inbound_Data.csv': Inbound_Data
        }

        # Radio button inside the expander
        data_source = st.radio("Choose Data Source:", list(data_files.keys()), index=0)

        # Store selection in session state and reset app if changed
        if st.session_state.get('current_data_source') != data_source:
            st.session_state.current_data_source = data_source
            reset_app_state()

    return api_key, data_files[data_source]


# ---------------------- UI Components ----------------------

def display_sample_data():
    """Display sample data in an expander."""
    with st.expander("📊 View Sample Data"):
        df = st.session_state.df.copy()
        for col in df.select_dtypes(include=['datetime64']):
            df[col] = df[col].dt.strftime('%d-%m-%Y')
        st.dataframe(df.head(), use_container_width=True)

#---------------------- Display Conversation UI ----------------------

def display_conversation_history():
    """Display all messages saved in session_state."""


    if "messages" in st.session_state.state:
        for msg in st.session_state.state["messages"]:
            # Here, msg.name can be used as the role identifier
            role = msg.name.upper() if msg.name else st.session_state.state["current"]
         
            # and msg.content holds the message text.
            # Convert Markdown in the content to HTML if needed.
            msg_html = markdown.markdown(msg.content)
            # Customize styling per role if desired.
            st.markdown(f"""
                <div style="background-color: #eaecee; padding: 10px; border-radius: 10px; margin: 10px 0;">
                    <strong style="color: #2a52be;">{role}:</strong>
                    <div style="color: #333;">{msg_html}</div>
                </div>
            """, unsafe_allow_html=True)



#---------------------- Main UI Function ----------------------

def process_conversation(config):
    """Processes the conversation state until it reaches FINISH or a counter limit."""


    counter = 0
    display_conversation_history()

    while st.session_state.state['next'] != 'FINISH' and counter < 10:

        if st.session_state.running_validate_parameters==False and st.session_state.running_await_user_validation==False:
            #Getting the current state
            st.session_state.current_state = multi_agent_graph.nodes[st.session_state.state['next']].invoke(st.session_state.state,config)
            ### Updating the state
            counter+=1
            st.session_state.state['current'] = st.session_state.state['next']



            msg = st.session_state.current_state['messages'][0].content
            msg_html = markdown.markdown(msg)  # Convert newlines to HTML line breaks

            st.markdown(f"""
                <div style="background-color: #eaecee; padding: 10px; border-radius: 10px; margin: 10px 0;">
                    <strong style="color: #2a52be;">{st.session_state.state['next'].upper()}:</strong>
                    <p style="color: #333;">{msg_html}</p>
                </div>
            """, unsafe_allow_html=True)



        if st.session_state.state['current'] in ['validate_parameters','await_user_validation']:
            if st.session_state.state['current']=='validate_parameters': # Asking user to chose `yes / no` ?
                if st.session_state.running_validate_parameters == False:
                    st.session_state.running_validate_parameters = True
                    st.stop()
                else:
                    # user_msg = input("Enter your response:")
                    st.session_state.current_state["messages"] = add_messages(st.session_state.current_state["messages"],[HumanMessage(content=st.session_state.validate_parameters_msg,name="Human")])
                    # st.session_state.current_state["messages"] = [HumanMessage(content=st.session_state.validate_parameters_msg,name="Human")]
                    st.session_state.running_validate_parameters =False
                    
            elif st.session_state.state['current']=='await_user_validation':
                if st.session_state.current_state['parameters_validated']==False:
                    if st.session_state.running_await_user_validation==False:
                        
                        st.session_state.running_await_user_validation = True
                        st.stop()
                    else:
                        # user_msg = input("Enter your response:")
                        st.session_state.current_state["messages"] = add_messages(st.session_state.current_state["messages"],[HumanMessage(content=st.session_state.await_user_validation_msg,name="Human")])
                        # st.session_state.current_state["messages"] = [HumanMessage(content=st.session_state.await_user_validation_msg,name="Human")]
                        st.session_state.running_await_user_validation = False
                else:
                    pass

        for key,item in st.session_state.current_state.items():
            if key=='messages':
                st.session_state.state[key] = add_messages(st.session_state.state[key],st.session_state.current_state[key]) # Key = "messages"
            else:
                st.session_state.state[key] = st.session_state.current_state[key]


    print("-"*30)
    print("State Info:\n")
    for k,v in st.session_state.state.items():
        if k=="messages":
            print("Messages till now>>")
            for msg in st.session_state.state["messages"]:
                print(f"{msg.name}: {msg.content}")
        elif k in ["current","next","pending_next","parameters_validated"]:
            print(k,v,sep=">>\n ")
        else:
            pass
    print("-"*30)

    if st.session_state.state['next']=="FINISH" and st.session_state.state["current"]=="supervisor":
        st.session_state.state["current"]="__start__"
        st.session_state.state["next"]="supervisor"
        print("***  Handle passed to the `__start__`  ***")
#################################################

    

    



#---------------------- Summarize Messages ----------------------

def summarize_messages(messages):
    """Summarizes a long conversation to maintain token limit."""
    prompt = "Summarize the following conversation in a concise way. While summarizing keep the facts alive, such as numeric values, parameters or specific results:\n\n"
    for msg in messages:
        prompt += f"{msg.content}\n"

    summary = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a helpful assistant that summarizes conversations."},
                  {"role": "user", "content": prompt}]
    )
    return summary.choices[0].message.content




#---------------------- Approximate Token Count ----------------------
def num_tokens_from_messages(messages, model="gpt-4o"):
    
    """Return the number of tokens used by a list of messages."""
    encoding = tiktoken.encoding_for_model(model)
    
    num_tokens = 0
    for message in messages:
        # Every message follows <im_start>{role/name}\n{content}<im_end>\n
        num_tokens += 4

        num_tokens += len(encoding.encode(message.content))
        num_tokens -= 1  # Role is always required and always 1 token
        num_tokens += 2  # Every reply is primed with <im_start>assistant
    
    return num_tokens


#---------------------- Main ----------------------

def main():
    """Main UI function to handle user interactions and execute the multi-agent graph."""
    st.title("UK Distribution CTS Insights & Optimisation Agent")
    api_key, data_file = setup_sidebar()
    os.environ["OPENAI_API_KEY"] = api_key

    if not api_key:
        st.info("Please enter your OpenAI API key in the sidebar to continue.")
        st.stop()

    if 'df' not in st.session_state:
        st.session_state.df = load_data_file(data_file)
        if st.session_state.df is None:
            st.stop()

    display_sample_data()


    if "bi_agent_responses" not in st.session_state:
        st.session_state.bi_agent_responses = []
    if 'cost_optimization_response' not in st.session_state:
        st.session_state.cost_optimization_response = []
    if 'static_optimization_response' not in st.session_state:
        st.session_state.static_optimization_response = []
    if 'running_validate_parameters' not in st.session_state:
        st.session_state.running_validate_parameters = False
    if 'running_await_user_validation' not in st.session_state:
        st.session_state.running_await_user_validation = False

    
    # Initialize conversation state if it doesn't exist
    if 'state' not in st.session_state or st.session_state.state is None:
        st.session_state.state = {"messages": [], "next": "supervisor","current":"__start__"}
 
    
    if 'current_state' not in st.session_state or st.session_state.current_state is None:
        st.session_state.current_state = {}


    
    # **Summarization Widget**
    st.sidebar.subheader("📝 Conversation Management")
    token_count = num_tokens_from_messages(st.session_state.state["messages"], "gpt-4o")
    st.sidebar.write(f"🟢 Approximate Token Count: **{token_count} / 100000**")
    
    if st.sidebar.button("Summarize Conversation 📝"):
        with st.status("🔄 Summarizing conversation..."):
            summary = summarize_messages(st.session_state.state["messages"])
            st.session_state.state["messages"] = [HumanMessage(content="Summary of past messages: " + summary,name="Summary")]
            st.success("Conversation summarized successfully.")


    # Sidebar Reset Button
    if st.sidebar.button("🔄 Reset Conversation"):
    # Clear only the necessary session state variables
        st.session_state.state = {"messages": [], "next": "supervisor","current":"__start__"}
        st.session_state.bi_agent_responses = []
        st.session_state.cost_optimization_response = []
        st.session_state.static_optimization_response = []
        st.session_state.running_validate_parameters = False
        st.session_state.running_await_user_validation = False
        
        # Ensure Streamlit reruns the script so history updates correctly
        st.rerun()



    # st.subheader("💬 GenAI Answer Bot")
    if user_question := st.chat_input("Type your message...", key="user_input"):

        # Append the message to conversation history
        st.session_state.state["messages"] = add_messages(
            st.session_state.state["messages"],
            [HumanMessage(content=user_question,name="Human")]
        )

        

        # Inject the response into the appropriate Streamlit variable for validation
        if st.session_state.running_validate_parameters:
            st.session_state.validate_parameters_msg = user_question
        elif st.session_state.running_await_user_validation:
            st.session_state.await_user_validation_msg = user_question

        # st.session_state.conversation_state["next"] = "supervisor"
        config = {"configurable": {"thread_id": "thread_id_1"}}



        graph_start_time = time.time()
        process_conversation(config)

        st.success("Processing complete.")
        st.markdown(
            f"""
                <div style="
                    background-color:#262730;
                    color:#00ffcc;
                    padding:10px;
                    border-radius:8px;
                    font-size:16px;
                    text-align:center;
                ">
                    ⏱️ <b>Analysis completed in {time.time() - graph_start_time:.1f} seconds</b>
                </div>
            """,
            unsafe_allow_html=True
        )

        

    # Display history in sidebar
    st.sidebar.subheader("🔍 History")
    st.write(" ")
    if st.session_state.bi_agent_responses:
        for i, response in enumerate(st.session_state.bi_agent_responses):
            with st.sidebar.expander(f"Query: {response['question'][:30]}...", expanded=False):
                st.markdown(f"**Question:** {response['question']}")
                st.markdown(f"**Time:** {response['timestamp']}")
                st.markdown("**Answer:**")
                st.markdown(response['answer'])

                if response['figure']:
                    st.image(response['figure'])

    if st.session_state.cost_optimization_response:
        for i, response in enumerate(st.session_state.cost_optimization_response):
            with st.sidebar.expander(f"Query: {response['query'][:30]}...", expanded=False):
                st.markdown(f"**Question:** {response['query']}")
                st.markdown(f"**Time:** {response['timestamp']}")
                st.markdown("**Answer:**")
                st.markdown(response['answer'])

    if st.session_state.static_optimization_response:
        for i, response in enumerate(st.session_state.static_optimization_response):
            with st.sidebar.expander(f"Query: {response['query'][:30]}...", expanded=False):
                st.markdown(f"**Question:** {response['query']}")
                st.markdown(f"**Time:** {response['timestamp']}")
                st.markdown("**Answer:**")
                st.markdown(response['answer'])

    if not st.session_state.bi_agent_responses and not st.session_state.cost_optimization_response and not st.session_state.static_optimization_response:
        st.sidebar.info("No responses yet.")



if __name__ == '__main__':
    main()