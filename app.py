import os
import openai
from openai import OpenAI
import streamlit as st
import pandas as pd
import random
import time
import re
from string import Template
from datetime import datetime

def get_openai_key():
    # Check if we're running on Streamlit Cloud
    is_streamlit_cloud = os.environ.get('STREAMLIT_RUNTIME_ENV') == 'cloud'
    
    if is_streamlit_cloud:
        # Use the secret key when hosted on Streamlit Cloud
        try:
            return st.secrets["OPENAI_API_KEY"]
        except KeyError:
            st.error("OpenAI API key not found in Streamlit secrets.")
            st.stop()
    else:
        # For local testing, use session state to store the API key
        if "openai_api_key" not in st.session_state:
            api_key = st.text_input("Enter your OpenAI API key for local testing:", type="password", key="openai_api_key_input")
            if api_key:
                st.session_state.openai_api_key = api_key
            else:
                st.warning("Please enter your OpenAI API key to proceed.")
                st.stop()
        return st.session_state.openai_api_key

# Initialize the OpenAI client
client = OpenAI(api_key=get_openai_key())
DEFAULT_MODEL = 'gpt-4o'  # Change to 'gpt-3.5-turbo' if needed

# New function to load VIP data from CSV
def load_vip_data(file_path='ContactsGPC.csv'):
    try:
        df = pd.read_csv(file_path)
        print("CSV Columns:", df.columns.tolist())  # Add this line
        return df
    except Exception as e:
        st.error(f"Error loading VIP data: {e}")
        return pd.DataFrame()

# Function to generate upcoming events (placeholder)
def generate_upcoming_events():
    events = [
        {"name": "Art Basel Miami Beach", "date": "December 1-3, 2023"},
        {"name": "Frieze Los Angeles", "date": "February 16-19, 2024"},
        {"name": "Venice Biennale", "date": "April 20 - November 24, 2024"},
        {"name": "Documenta", "date": "June 18 - September 25, 2022"},
        {"name": "Art Basel Hong Kong", "date": "March 28-30, 2024"}
    ]
    return random.sample(events, 3)

INSIGHTS_PROMPT_TEMPLATE = Template("""
Analyze the following VIP client's data and provide insights:

Name: $name
Purchase History: $purchase_history
Interaction History: $interaction_history
Preferred Contact Times: $preferred_contact_times
Last Contact Date: $last_contact_date
Sentiment Score: $sentiment_score

Provide suggestions on how to best engage with this client.
""")

MESSAGE_PROMPT_TEMPLATE = Template("""
Compose a personalized invitation email to $name for the upcoming Art Basel event.
Mention their interest in $purchase_history and reference their previous interaction: $interaction_history.
Suggest scheduling a meeting during their preferred contact time: $preferred_contact_times.
""")

SENTIMENT_PROMPT_TEMPLATE = Template("""
Analyze the sentiment of the following interaction history and provide a score between -1 (very negative) and 1 (very positive):

$interaction_history

Please respond with only a number between -1 and 1, representing the sentiment score.
""")

def generate_openai_response(prompt, max_tokens=250, temperature=0.7, retries=3):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                n=1
            )
            # Validate response structure
            if hasattr(response, 'choices') and len(response.choices) > 0:
                return response.choices[0].message.content.strip()
            else:
                st.error(f"Unexpected API response structure: {response}")
                return ""
        except Exception as e:
            st.error(f"Error during OpenAI API call: {str(e)}")
            if attempt < retries - 1:
                st.warning(f"Retrying... (Attempt {attempt + 2} of {retries})")
            else:
                return ""

def generate_vip_insights(vip_info, name_column, additional_info=""):
    prompt = INSIGHTS_PROMPT_TEMPLATE.substitute(
        name=vip_info.get(name_column, "Unknown"),
        purchase_history=vip_info.get('Purchase History', "Not available"),
        interaction_history=vip_info.get('Interaction History', "Not available"),
        preferred_contact_times=vip_info.get('Preferred Contact Times', "Not available"),
        last_contact_date=vip_info.get('Last Contact Date', "Not available"),
        sentiment_score=vip_info.get('Sentiment Score', "Not available")
    )
    
    if additional_info:
        prompt += f"\n\nAdditional information from web search:\n{additional_info}\n\nPlease incorporate this information into your insights and suggestions."
    
    return generate_openai_response(prompt)

def generate_personalized_message(vip_info, name_column):
    prompt = MESSAGE_PROMPT_TEMPLATE.substitute(
        name=vip_info.get(name_column, "Unknown"),
        purchase_history=vip_info.get('Purchase History', "Not available"),
        interaction_history=vip_info.get('Interaction History', "Not available"),
        preferred_contact_times=vip_info.get('Preferred Contact Times', "Not available")
    )
    return generate_openai_response(prompt)

def analyze_sentiment(vip_info):
    prompt = SENTIMENT_PROMPT_TEMPLATE.substitute(
        interaction_history=vip_info.get('Interaction History', "Not available")
    )
    sentiment_response = generate_openai_response(prompt, max_tokens=100, temperature=0)
    
    # Try to extract a numerical value from the response
    numerical_match = re.search(r'(-?\d+(\.\d+)?)', sentiment_response)
    if numerical_match:
        try:
            sentiment_score = float(numerical_match.group(1))
            if -1 <= sentiment_score <= 1:
                return sentiment_score
            else:
                st.warning(f"Extracted sentiment score {sentiment_score} is out of expected range [-1, 1].")
        except ValueError:
            pass
    
    # If no valid numerical value found, estimate based on the text
    lower_response = sentiment_response.lower()
    if 'positive' in lower_response:
        return 0.5
    elif 'negative' in lower_response:
        return -0.5
    elif 'neutral' in lower_response:
        return 0
    else:
        st.warning(f"Unable to determine sentiment score from response: '{sentiment_response}'")
        return 0  # Default to neutral

def get_engagement_score():
    # Simulated engagement score between 50 and 100
    return random.randint(50, 100)

# New functions
def generate_vip_summary(vip_info, name_column):
    prompt = f"Create a concise 3-4 sentence summary of the VIP named {vip_info.get(name_column, 'Unknown')} based on this information: {vip_info}. Highlight key interests and important points for engagement."
    return generate_openai_response(prompt)

def generate_engagement_suggestions(vip_info, name_column):
    prompt = f"Based on the VIP {vip_info.get(name_column, 'Unknown')}'s profile ({vip_info}), suggest 3 personalized ways to engage with them."
    return generate_openai_response(prompt)

def recommend_events(vip_info, upcoming_events, name_column):
    events_str = ", ".join([f"{e['name']} on {e['date']}" for e in upcoming_events])
    prompt = f"Given the VIP {vip_info.get(name_column, 'Unknown')}'s background ({vip_info}), which of these upcoming events would they likely be interested in: {events_str}? Explain why for the top 2 recommendations."
    return generate_openai_response(prompt)

def generate_conversation_starters(vip_info, name_column):
    prompt = f"Create 3 engaging conversation starters for a VIP named {vip_info.get(name_column, 'Unknown')} with this background: {vip_info}."
    return generate_openai_response(prompt)

def plan_follow_up_actions(vip_info, interaction_notes, name_column):
    prompt = f"Based on the VIP {vip_info.get(name_column, 'Unknown')}'s profile ({vip_info}) and recent interaction notes ({interaction_notes}), suggest 3 follow-up actions for the next 2 weeks."
    return generate_openai_response(prompt)

def curate_personalized_content(vip_info, name_column):
    prompt = f"Suggest 5 recent articles, videos, or artworks that would interest a VIP named {vip_info.get(name_column, 'Unknown')} with this background: {vip_info}."
    return generate_openai_response(prompt)

def simulated_web_search(name, context="art collector"):
    current_year = datetime.now().year
    prompt = f"""
    Simulate a web search for {name} in the context of being an {context}. 
    Provide a summary of what might be found online about this person's involvement in the art world, 
    their notable collections, or contributions to art. 
    Include possible details such as:
    - Their background and how they got into art collecting
    - Any famous pieces they might own
    - Their focus or specialty in art collecting
    - Any public exhibitions or donations they might have made
    - Their influence in the art world

    Remember, this is a simulation based on the name provided. The information generated 
    should be plausible but may not be factual. The current year is {current_year}, 
    and your knowledge cutoff is 2022, so avoid mentioning specific recent events.

    Begin your response with: "Based on a simulated web search, here's what we might find about {name} as an art collector:"
    """
    return generate_openai_response(prompt, max_tokens=300)

# Main app
def main():
    st.title("Art Basel AI-Driven CRM Prototype")
    
    # Debug information
    st.sidebar.write("Debug Information:")
    api_key = get_openai_key()
    st.sidebar.write(f"API Key (first 5 chars): {api_key[:5]}...")
    
    # Test API connection
    try:
        test_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, World!"}],
            max_tokens=5
        )
        st.sidebar.success("API connection test successful!")
    except Exception as e:
        st.sidebar.error(f"API connection test failed: {str(e)}")

    # Load VIP Data
    vip_data = load_vip_data()
    
    if vip_data.empty:
        st.error("No data loaded. Please check your CSV file.")
        return

    # Check for the correct column name
    name_column = 'Full Name'
    if name_column not in vip_data.columns:
        possible_name_columns = [col for col in vip_data.columns if 'name' in col.lower()]
        if possible_name_columns:
            name_column = possible_name_columns[0]
            st.warning(f"'Full Name' column not found. Using '{name_column}' instead.")
        else:
            st.error("Could not find a suitable name column. Please check your CSV file.")
            return

    # Sidebar for VIP Selection
    selected_vip = st.sidebar.selectbox("Select VIP Client", vip_data[name_column].tolist())

    # Retrieve VIP Information
    vip_info = vip_data[vip_data[name_column] == selected_vip].iloc[0].to_dict()

    # Display VIP Profile
    st.header(f"Profile: {vip_info[name_column]}")
    
    # Display other fields if they exist
    for field in ['Email 1', 'Main Address: City', 'Main Address: Country', 'Company Name', 'Job Title', 'Industry', 'Biography']:
        if field in vip_info:
            st.write(f"**{field}:** {vip_info[field]}")
        else:
            st.write(f"**{field}:** Not available")

    # Predictive Engagement Score
    engagement_score = get_engagement_score()
    st.subheader("Predictive Engagement Score")
    st.progress(engagement_score)
    st.write(f"Engagement Likelihood: {engagement_score}%")

    # Simulated Web Search for Additional Information
    if st.button("Simulate Web Search for Additional Info"):
        with st.spinner("Simulating a web search for additional information..."):
            additional_info = simulated_web_search(vip_info[name_column])
        st.subheader("Simulated Additional Information")
        st.write(additional_info)
        st.warning("Note: This information is simulated and may not be accurate or up-to-date.")

        # Store the additional info in session state
        st.session_state.additional_info = additional_info

    # Generate AI Insights (update this to use the simulated info)
    if st.button("Generate AI Insights"):
        with st.spinner("Generating insights..."):
            additional_info = st.session_state.get('additional_info', '')
            insights = generate_vip_insights(vip_info, name_column, additional_info)
        if insights:
            st.subheader("AI-Generated Insights")
            st.write(insights)

    # Generate Personalized Message
    if st.button("Generate Personalized Message"):
        with st.spinner("Generating message..."):
            message = generate_personalized_message(vip_info, name_column)
        if message:
            st.subheader("Personalized Message")
            st.write(message)

    # Analyze Sentiment
    if st.button("Analyze Sentiment"):
        with st.spinner("Analyzing sentiment..."):
            sentiment = analyze_sentiment(vip_info)
        if sentiment is not None:
            st.subheader("Sentiment Analysis")
            st.write(f"Sentiment Score: {sentiment:.2f}")
        else:
            st.error("Unable to determine sentiment score.")

    # VIP Summary
    if st.button("Generate VIP Summary"):
        with st.spinner("Generating VIP summary..."):
            summary = generate_vip_summary(vip_info, name_column)
        st.subheader("VIP Summary")
        st.write(summary)

    # Engagement Suggestions
    if st.button("Generate Engagement Suggestions"):
        with st.spinner("Generating engagement suggestions..."):
            suggestions = generate_engagement_suggestions(vip_info, name_column)
        st.subheader("Engagement Suggestions")
        st.write(suggestions)

    # Event Recommendations
    upcoming_events = generate_upcoming_events()
    if st.button("Recommend Events"):
        with st.spinner("Generating event recommendations..."):
            recommendations = recommend_events(vip_info, upcoming_events, name_column)
        st.subheader("Event Recommendations")
        st.write(recommendations)

    # Conversation Starters
    if st.button("Generate Conversation Starters"):
        with st.spinner("Generating conversation starters..."):
            starters = generate_conversation_starters(vip_info, name_column)
        st.subheader("Conversation Starters")
        st.write(starters)

    # Follow-up Action Planner
    interaction_notes = st.text_area("Enter recent interaction notes:")
    if st.button("Plan Follow-up Actions"):
        with st.spinner("Planning follow-up actions..."):
            actions = plan_follow_up_actions(vip_info, interaction_notes, name_column)
        st.subheader("Follow-up Action Plan")
        st.write(actions)

    # Personalized Content Curator
    if st.button("Curate Personalized Content"):
        with st.spinner("Curating personalized content..."):
            content = curate_personalized_content(vip_info, name_column)
        st.subheader("Personalized Content Recommendations")
        st.write(content)

if __name__ == "__main__":
    main()