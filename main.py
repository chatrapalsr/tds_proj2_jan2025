import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import openai

# Set your OpenAI API key in your environment variables.
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise Exception("Please set the OPENAI_API_KEY environment variable")

app = FastAPI()

# Enable CORS for all origins (this is important for public APIs)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

def determine_assignment(question: str) -> str:
    """
    Identify the assignment type from the question.
    Customize this function as needed to better classify questions.
    """
    if "GA 1" in question:
        return "GA 1"
    elif "GA 2" in question:
        return "GA 2"
    elif "GA 3" in question:
        return "GA 3"
    elif "GA 4" in question:
        return "GA 4"
    elif "GA 5" in question:
        return "GA 5"
    else:
        return "General"

def generate_llm_answer(question: str, assignment: str, files: Optional[List[UploadFile]]) -> str:
    """
    Generates an answer using OpenAI's API.
    Reinforce the LLM logic by including clear context on the assignment type and question.
    """
    # Build a detailed prompt that helps the LLM focus on the assignment
    prompt = f"Assignment Type: {assignment}\nQuestion: {question}\n"
    
    # If any files are attached, include some content from them in the prompt
    if files:
        for file in files:
            content = file.file.read().decode("utf-8", errors="replace")
            prompt += f"\nContent from {file.filename}:\n{content}\n"
            file.file.seek(0)  # Reset pointer for any further use

    # Call the OpenAI API
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # You can use "gpt-4" if available and desired
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant that provides concise answers for graded assignment questions. Your output must be exactly in JSON format: {\"answer\": \"<the answer>\"} with no extra text."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2  # Lower temperature for more deterministic responses
        )
        # The LLM answer should ideally be formatted as the JSON we expect.
        answer = response.choices[0].message["content"].strip()
        return answer
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/")
async def answer_question(
    question: str = Form(...),
    file: Optional[List[UploadFile]] = File(None)
):
    assignment = determine_assignment(question)
    answer = generate_llm_answer(question, assignment, file)
    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
