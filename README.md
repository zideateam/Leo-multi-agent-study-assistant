# 🦁 Leo — Multi-Agent Study Assistant

Leo is a **CrewAI-based multi-agent study assistant**. A student chooses a topic and learning level, then several specialized agents collaborate to plan the lesson, teach it, create a structured quiz, evaluate the student's answers, and give final learning guidance.

The project is designed to run in **Google Colab** and can also run locally.

---

## Features

- 4 distinct AI agent roles
- Real agent-to-agent handoffs
- Sequential orchestration
- Structured quiz output using Pydantic
- Structured evaluation output using Pydantic
- Student/topic memory
- Gradio web interface
- Error handling for invalid requests and failed agent execution
- Async CrewAI execution for Google Colab / Jupyter compatibility
- No API keys stored in the repository

---

## Agents and Roles

### 1. Coordinator

**Role:** `Leo Learning Coordinator`

The Coordinator receives the student's name, topic, level, and previous score.

Responsibilities:

- understand the student's learning request
- create the learning plan
- define important concepts
- determine a suitable teaching sequence
- guide the Explainer and Quiz Master
- provide final study advice after evaluation
- narrow overly broad topics instead of failing

---

### 2. Explainer

**Role:** `Leo Concept Explainer`

Responsibilities:

- receive the Coordinator's learning plan
- teach the requested concept
- adapt the lesson to the student's level
- use examples and analogies
- identify common mistakes
- provide a short summary

The Explainer receives the Coordinator task through CrewAI task context.

---

### 3. Quiz Master

**Role:** `Leo Quiz Master`

Responsibilities:

- receive the Coordinator's plan
- receive the Explainer's lesson
- create exactly five multiple-choice questions
- create four options for every question
- return the quiz in structured Pydantic output

The Quiz Master does not work independently. Its quiz is based on the outputs of the previous agents.

---

### 4. Evaluator

**Role:** `Leo Student Answer Evaluator`

Responsibilities:

- receive the exact Quiz Master's quiz
- receive the student's answers
- compare answers with the correct-answer fields
- calculate score and percentage
- identify strengths and weaknesses
- produce feedback and a learning recommendation

The Evaluator output is then passed back to the Coordinator for final learning guidance.

---

## Architecture

```text
                         ┌───────────────────────┐
                         │        Student        │
                         │ Name / Topic / Level  │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │    Request Validator  │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ Coordinator Agent     │
                         │ Creates learning plan │
                         └───────────┬───────────┘
                                     │
                              Task Context
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ Explainer Agent       │
                         │ Teaches the lesson    │
                         └───────────┬───────────┘
                                     │
                              Task Context
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ Quiz Master Agent     │
                         │ Pydantic QuizPack     │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │        Student        │
                         │ Answers A,B,C,D,A     │
                         └───────────┬───────────┘
                                     │
                     QuizPack + Student Answers
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ Evaluator Agent       │
                         │ Score + Feedback      │
                         └───────────┬───────────┘
                                     │
                              Task Context
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ Coordinator Agent     │
                         │ Final study advice    │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │ Student Memory JSON   │
                         └───────────────────────┘
```

---

## Orchestration Pattern

Leo uses a **sequential orchestration pattern**.

### Phase 1 — Teaching

```text
Coordinator
    ↓
Explainer
    ↓
Quiz Master
    ↓
Student
```

CrewAI executes the tasks with:

```python
process=Process.sequential
```

The Explainer receives the Coordinator output through:

```python
context=[planning_task]
```

The Quiz Master receives both previous outputs:

```python
context=[
    planning_task,
    explanation_task
]
```

This demonstrates a real handoff rather than isolated agent responses.

### Phase 2 — Evaluation

```text
Student Answers
      ↓
Evaluator
      ↓
Coordinator
      ↓
Final Guidance
```

The generated `QuizPack` and student answers are passed to the Evaluator. The Evaluator result is then passed through task context to the Coordinator.

---

## Structured Output

The Quiz Master returns a Pydantic model:

```python
class QuizPack(BaseModel):
    topic: str
    questions: List[QuizQuestion]
```

Each question contains:

```text
question_number
question
options
correct_answer
explanation
```

The Evaluator also returns a structured model containing:

```text
score
total_questions
percentage
correct_questions
incorrect_questions
strengths
weaknesses
feedback
recommendation
```

---

## Memory

Leo uses two forms of memory.

### CrewAI memory

Both crews are configured with:

```python
memory=True
```

### Application-level student memory

The application stores previous student sessions in:

```text
data/leo_student_memory.json
```

This runtime file is excluded from Git by `.gitignore`.

Stored data includes:

- student name
- last topic
- learning level
- previous score
- session history

---

## Project Structure

```text
leo-multi-agent-study-assistant/
│
├── app.py
├── leo_core.py
├── memory_store.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
│
└── data/
    └── .gitkeep
```

### `app.py`

Gradio interface and async application workflow.

### `leo_core.py`

Contains:

- Pydantic schemas
- LLM configuration
- four CrewAI agent definitions
- teaching crew
- evaluation crew
- validation helpers
- quiz formatting helpers

### `memory_store.py`

Handles student session memory.

---

# Run in Google Colab

## 1. Clone the repository

```python
!git clone https://github.com/YOUR_USERNAME/leo-multi-agent-study-assistant.git
%cd leo-multi-agent-study-assistant
```

---

## 2. Install packages

```python
!pip -q install -r requirements.txt
```

If Colab reports dependency warnings from unrelated preinstalled packages such as Google ADK, BigFrames, or Tweepy, first verify that the required Leo packages import successfully before changing unrelated dependencies.

---

## 3. Add OpenAI API Key to Colab Secrets

In Google Colab:

```text
Left sidebar
→ Secrets 🔑
→ Add new secret
```

Create:

```text
Name: OPENAI_API_KEY
Value: your real OpenAI API key
```

Enable **Notebook access**.

Do not write the real key directly into the notebook.

---

## 4. Load the secret

Run:

```python
import os
from google.colab import userdata

os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")

print("✅ API key loaded")
```

---

## 5. Start the application

Because `app.py` already uses async CrewAI callbacks, run:

```python
!python app.py
```

For Colab, if you want a public Gradio URL, change the last line in `app.py` from:

```python
demo.launch()
```

to:

```python
demo.launch(share=True, debug=True)
```

Then rerun:

```python
!python app.py
```

Open the generated `gradio.live` link.

---

# Run Locally

Requires Python 3.10–3.13 for CrewAI 1.15.44.

## 1. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

---

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Set the API key

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your-key"
```

Linux/macOS:

```bash
export OPENAI_API_KEY="your-key"
```

---

## 4. Start Leo

```bash
python app.py
```

Open the local Gradio URL shown in the terminal.

---

# API Key Security

This repository intentionally contains **no API key**.

The `.gitignore` file excludes:

```text
.env
.env.*
```

The repository includes only:

```text
.env.example
```

with a placeholder value.

Before pushing to GitHub, check:

```bash
git status
```

and make sure `.env` or any notebook containing a real key is not staged.

Never commit code such as:

```python
OPENAI_API_KEY = "sk-..."
```

---

# Example Workflow

Student enters:

```text
Name: Zubair
Topic: Python Dictionaries
Level: Beginner
```

Leo runs:

```text
Coordinator
  ↓ creates plan

Explainer
  ↓ teaches lesson

Quiz Master
  ↓ creates validated five-question quiz

Student
  ↓ submits A,B,C,D,A

Evaluator
  ↓ calculates score and feedback

Coordinator
  ↓ gives final study recommendation

Memory
  ↓ stores session
```

---

# Error Handling

Leo handles several common problems.

### Missing student information

The request validator blocks incomplete submissions.

### Invalid answer format

The expected format is:

```text
A,B,C,D,A
```

Only `A`, `B`, `C`, or `D` are accepted.

### Agent/LLM failure

Agent execution is wrapped in `try/except` blocks and the UI displays a Coordinator-style error message rather than crashing.

### Colab event-loop issue

Crew execution uses:

```python
await crew.kickoff_async(...)
```

instead of synchronous `crew.kickoff(...)`.

This avoids the common Jupyter/Colab error:

```text
Agent execution was invoked synchronously from within a running event loop.
```

---

# Technologies

- Python
- CrewAI
- OpenAI API
- Gradio
- Pydantic
- Google Colab

---

# Assignment Requirements Mapping

| Requirement | Implementation |
|---|---|
| Coordinator | `Leo Learning Coordinator` |
| Explainer | `Leo Concept Explainer` |
| Quiz Master | `Leo Quiz Master` |
| Evaluator | `Leo Student Answer Evaluator` |
| Distinct roles/prompts | Separate CrewAI Agent definitions |
| Agent handoff | CrewAI `Task.context` + structured QuizPack transfer |
| Structured quiz | Pydantic |
| Orchestration | `Process.sequential` |
| Memory | CrewAI memory + JSON student memory |
| Graceful errors | validation + `try/except` |
| Interface | Gradio |
| Agent activity visible | status panel in UI |
| API keys protected | environment variable + `.gitignore` |

---

## License

This project is intended as an educational assignment/demo. Add the license required by your course or organization before redistribution.
