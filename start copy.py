# Clarify with user what there ideas is
# Agents running to get predefined questions based on users idea
# Summarize info in form of report

import asyncio
import aiohttp
from openai import AsyncOpenAI

api_key = ""
async def llm_call(prompt):
    print(prompt)
    client = AsyncOpenAI(api_key=api_key)
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a research assistant"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )
    output = response.choices[0].message.content
    print(output)
    return output

async def startup_idea_summarize(answers):
    prompt = "Based on answers " + ' '.join(answers) + " summarize a startup idea with comprehensive details"
    return await llm_call(prompt)

async def clarification_questions():
    questions = [
        "Describe your start-up idea in one or two sentences."#,
        # "What specific problem does your start-up solve?",
        # "Who are your target customers, and how does the problem impact them?",
        # "What makes your solution unique?"
    ]
    answers = []
    for question in questions:
        answer = input(question + " ")  # Still blocking, consider an async alternative if using in a web app
        answers.append(answer)

    return await startup_idea_summarize(answers)    

async def fetch_data(client, prompt, idea):
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt.replace("[Insert Start-up Idea]", idea)}
            ],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

async def analyze_startup(idea):
    prompts = {
        "Problem-Solution Fit": "Analyze this start-up idea: [Insert Start-up Idea]. What specific problem is it solving? Who experiences this problem, and how painful is it?",
        "Market Analysis": "Evaluate the market potential for a start-up focused on [Insert Start-up Idea]. Who are the ideal customers, what is the market size, and are there any emerging trends?",
        "Competitive Landscape": "Identify potential competitors for [Insert Start-up Idea]. How does this idea differentiate from existing solutions?",
        "Business Model & Revenue Streams": "Suggest a viable business model for [Insert Start-up Idea]. How can it generate revenue?",
        "Go-To-Market Strategy": "Propose a go-to-market strategy for [Insert Start-up Idea]. Who are the early adopters and what marketing channels should be used?",
        "Product Feasibility & Scalability": "Evaluate the technical feasibility of [Insert Start-up Idea]. What are the key technical challenges?",
        "Team & Execution": "What skills and expertise are essential for successfully executing [Insert Start-up Idea]?",
        "Financial Projections & Funding": "Estimate the initial funding requirements for [Insert Start-up Idea]. What major costs should be considered?",
        "Risk Assessment & Mitigation": "List the potential risks involved in launching [Insert Start-up Idea]. Suggest mitigation strategies.",
        "SWOT Analysis": "Create a SWOT analysis (Strengths, Weaknesses, Opportunities, Threats) for [Insert Start-up Idea].",
        "Pitch Deck Suggestions": "Outline a pitch deck for [Insert Start-up Idea]. Include slides for Problem, Solution, Market Opportunity, Business Model, and Go-to-Market Strategy.",
        "Start-Up Idea Scoring": "Score the following start-up idea based on Market Potential, Product Feasibility, Competitive Advantage, and Revenue Model. Provide a score from 1-10 for each category."
    }
    
    client = AsyncOpenAI(api_key=api_key)
    tasks = [fetch_data(client, prompt, idea) for _, prompt in prompts.items()]
    results = await asyncio.gather(*tasks)

    return {category: result for category, result in zip(prompts.keys(), results)}

def generate_report(idea, insights):
    report = f"Startup Analysis Report for: {idea}\n\n"
    for section, content in insights.items():
        report += f"### {section}\n{content}\n\n"
    return report

async def main():
    print("Welcome to the Startup Idea Analyzer!")
    idea = clarification_questions()
    insights = await analyze_startup(idea)
    report = generate_report(idea, insights)
    print(report)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())
