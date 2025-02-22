from openai import OpenAI

def llm_call(prompt, api_key):
    print(prompt)
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
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

def startup_idea_summarize(answers, api_key):
    prompt = "Based on answers " + ' '.join(answers) + " summarize to give final startup idea"
    return llm_call(prompt, api_key)

def clarification_questions(api_key):
    questions = [
        "Describe your start-up idea in one or two sentences."
    ]
    answers = []
    for question in questions:
        answer = input(question + " ")
        answers.append(answer)

    return startup_idea_summarize(answers, api_key)

def fetch_data(client, prompt, idea):
    try:
        response = client.chat.completions.create(
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

def analyze_startup(idea, api_key):
    prompts = {
        "Problem-Solution Fit": "Analyze this start-up idea: [Insert Start-up Idea]. What specific problem is it solving? Who experiences this problem, and how painful is it?",
        "Market Analysis": "Evaluate the market potential for a start-up focused on [Insert Start-up Idea]. Who are the ideal customers, what is the market size, and are there any emerging trends?",
        "Competitive Landscape": "Identify potential competitors for [Insert Start-up Idea]. How does this idea differentiate from existing solutions?"#,
        # "Business Model & Revenue Streams": "Suggest a viable business model for [Insert Start-up Idea]. How can it generate revenue?",
        # "Go-To-Market Strategy": "Propose a go-to-market strategy for [Insert Start-up Idea]. Who are the early adopters and what marketing channels should be used?",
        # "Product Feasibility & Scalability": "Evaluate the technical feasibility of [Insert Start-up Idea]. What are the key technical challenges?",
        # "Team & Execution": "What skills and expertise are essential for successfully executing [Insert Start-up Idea]?",
        # "Financial Projections & Funding": "Estimate the initial funding requirements for [Insert Start-up Idea]. What major costs should be considered?",
        # "Risk Assessment & Mitigation": "List the potential risks involved in launching [Insert Start-up Idea]. Suggest mitigation strategies.",
        # "SWOT Analysis": "Create a SWOT analysis (Strengths, Weaknesses, Opportunities, Threats) for [Insert Start-up Idea].",
        # "Pitch Deck Suggestions": "Outline a pitch deck for [Insert Start-up Idea]. Include slides for Problem, Solution, Market Opportunity, Business Model, and Go-to-Market Strategy.",
        # "Start-Up Idea Scoring": "Score the following start-up idea based on Market Potential, Product Feasibility, Competitive Advantage, and Revenue Model. Provide a score from 1-10 for each category."
    }
    
    client = OpenAI(api_key=api_key)
    results = {}
    for category, prompt in prompts.items():
        results[category] = fetch_data(client, prompt, idea)
        print(f"{category}: {results[category]}")
        print("--------------------------------------------------")

    return results

def generate_report(idea, insights):
    report = f"Startup Analysis Report for: {idea}\n\n"
    for section, content in insights.items():
        report += f"### {section}\n{content}\n\n"
    return report

def main():
    api_key = ''
    print("Welcome to the Startup Idea Analyzer!")
    looks_good = False
    while not looks_good:
        idea = clarification_questions(api_key)
        print("Analyzing your startup idea... " + idea)
        yes_or_no = input("Is this correct? (yes/no) ")
        if yes_or_no == "yes":
            looks_good = True
    
    insights = analyze_startup(idea, api_key)
    print(insights)
    # report = generate_report(idea, insights)
    # print(report)

if __name__ == "__main__":
    main()
