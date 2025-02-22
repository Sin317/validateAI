import gradio as gr
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama.llms import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage
from llama_index.llms.ollama import Ollama
import json
import numpy as np
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.ollama import OllamaEmbedding
from phoenix.evals import LiteLLMModel
import phoenix as px
import pandas as pd


# Initialize Ollama model
llm = OllamaLLM(model="llama3.2", num_gpu=1) 
conversation_history = []


history = []
# Load stored embeddings for startup ideas
with open("embeddings_startups_1k.json", "r") as f:
    embedding_data = json.load(f)

# Convert embeddings to NumPy arrays for similarity search
texts = [item["text"] for item in embedding_data]
embeddings = np.array([item["embedding"] for item in embedding_data])

# Function to find similar startups based on query
def query_startup_ideas(query: str):
    query_embedding = OllamaEmbedding(model_name="llama3.2").get_text_embedding(query)

    # Compute cosine similarity
    similarities = np.dot(embeddings, query_embedding) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding))

    # Get top 3 matches
    top_indices = np.argsort(similarities)[::-1][:3]
    results = [texts[i] for i in top_indices]

    return "\n\n".join(results)

# Define the prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Startup Idea Validator. Start by asking the user for their idea and continue asking questions until you fully understand what they want to do. Once confirmed, provide in-depth analysis of their startup idea feasibility and market research."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

# Initialize the LLMChain
chain = prompt_template | llm

# Define the function to generate the startup report
def generate_report(idea, insights, similar_startups):
    report = f"Startup Analysis Report for: {idea}\n\n"
    
    # Add similar startups
    report += f"### Similar Startups\n{similar_startups}\n\n"

    # Add analysis insights
    for section, content in insights.items():
        report += f"### {section}\n{content}\n\n"
    
    return report

# Define the function for Gradio interface
def gradio_interface(user_input, conversation_history=None, confirmation=False):
    """
    Function to handle the logic when the application interface is invoked with a prompt.
    :param user_input: the prompt passed by the user.
    :param conversation_history: a list containing the user and model's interactions for context.
    """
    # Build context for the model
    if conversation_history is None:
        conversation_history = []
    
    
    # Get the response from the LLM
    response = chain.invoke({"input": user_input, "chat_history": conversation_history})
    
    # Add the user prompt and AI response to history
    conversation_history.append(HumanMessage(content=user_input))
    conversation_history.append(AIMessage(content=response))
    
    history.append({"role": "user", "content":user_input})
    
    history.append({"role": "assistant", "content": response})


    if not confirmation:
        return history, "Please confirm first", "Please confirm first", conversation_history

    # Query similar startups
    similar_startups = query_startup_ideas(user_input)

    # Analyze the startup idea
    prompts = {
        "Problem-Solution Fit": "Analyze this start-up idea: {idea}. What specific problem is it solving? Who experiences this problem, and how painful is it?",
        "Market Analysis": "Evaluate the market potential for a start-up focused on {idea}. Who are the ideal customers, what is the market size, and are there any emerging trends?",
        # "Competitive Landscape": "Identify potential competitors for {idea}. How does this idea differentiate from existing solutions?",
        # "Business Model & Revenue Streams": "Suggest a viable business model for {idea}. How can it generate revenue?",
        # "Go-To-Market Strategy": "Propose a go-to-market strategy for {idea}. Who are the early adopters and what marketing channels should be used?",
        # "Product Feasibility & Scalability": "Evaluate the technical feasibility of {idea}. What are the key technical challenges?",
        # "Team & Execution": "What skills and expertise are essential for successfully executing {idea}?",
        # "Financial Projections & Funding": "Estimate the initial funding requirements for {idea}. What major costs should be considered?",
        # "Risk Assessment & Mitigation": "List the potential risks involved in launching {idea}. Suggest mitigation strategies.",
        # "SWOT Analysis": "Create a SWOT analysis (Strengths, Weaknesses, Opportunities, Threats) for {idea}.",
        # "Pitch Deck Suggestions": "Outline a pitch deck for {idea}. Include slides for Problem, Solution, Market Opportunity, Business Model, and Go-to-Market Strategy.",
        "Start-Up Idea Scoring": "Score the following start-up idea based on Market Potential, Product Feasibility, Competitive Advantage, and Revenue Model. Provide a score from 1-10 for each category."
    }

    # Analyzing the startup idea using Ollama
    analysis_results = {}
    for category, prompt in prompts.items():
        response = llm.invoke(prompt.format(idea=user_input))  # Using Ollama to fetch insights
        analysis_results[category] = response

    # Generate the report
    report = generate_report(user_input, analysis_results, similar_startups)
    # report = "test output"

    px.launch_app()
    model = LiteLLMModel(model_name="llama3.2")

    eval_results = []
    for message in history:
        if message["role"] == "user":
            user_message = message["content"]
        elif message["role"] == "assistant":
            assistant_message = message["content"]
            eval_result = model.score(
                prompt=user_message,
                response=assistant_message,
                metrics=["relevance", "coherence"]
            )
            eval_results.append(eval_result)

    df = pd.DataFrame(eval_results)
    px.log_dataframe("evaluation_results", df)
    return history, similar_startups, report, conversation_history

# Initialize Gradio app
def init_app():
    input_textbox = gr.Textbox(label="User Prompt", placeholder="Enter your startup idea", lines=2)
    output_textbox = gr.Chatbot(type="messages")
    similar_startups_output = gr.Textbox(label="Similar Startups", placeholder="Similar startups based on your idea", lines=5)
    report_output = gr.Textbox(label="Startup Analysis Report", placeholder="Generated report for your startup idea", lines=10)
    confirmation_checkbox = gr.Checkbox(label="Confirm that the LLM understands your idea correctly")
    app = gr.Interface(
       fn=gradio_interface,
       inputs=[input_textbox, gr.State(), confirmation_checkbox],
       outputs=[output_textbox, similar_startups_output, report_output, gr.State()],
       title="Chat with Startup Idea Validator",
       live=False
   )

    app.launch()

# Run the app
if __name__ == "__main__":
    init_app()
