from __future__ import annotations

import json
import os
from typing import List

from crewai import Agent, Crew, LLM, Process, Task
from pydantic import BaseModel, Field


class QuizQuestion(BaseModel):
    question_number: int = Field(description="Question number starting from 1")
    question: str
    options: List[str] = Field(description="Exactly four answer choices")
    correct_answer: str = Field(description="Correct option letter: A, B, C, or D")
    explanation: str


class QuizPack(BaseModel):
    topic: str
    questions: List[QuizQuestion] = Field(description="Exactly five quiz questions")


class EvaluationResult(BaseModel):
    score: int
    total_questions: int
    percentage: float
    correct_questions: List[int]
    incorrect_questions: List[int]
    strengths: List[str]
    weaknesses: List[str]
    feedback: str
    recommendation: str


def validate_environment() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "In Colab, load it from Secrets. "
            "Locally, export it as an environment variable."
        )


def build_llm() -> LLM:
    validate_environment()
    return LLM(
        model="openai/gpt-4o",
        temperature=0.3,
    )


def build_agents(llm: LLM):
    coordinator = Agent(
        role="Leo Learning Coordinator",
        goal=(
            "Understand the student's request, create an appropriate learning plan, "
            "coordinate the tutoring workflow, and provide final learning guidance."
        ),
        backstory=(
            "You lead Leo, a collaborative multi-agent study assistant. "
            "You analyze the student's topic, level, and previous performance. "
            "You create a clear plan for the Explainer and Quiz Master. "
            "If a request is vague, narrow it to a reasonable introductory lesson."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )

    explainer = Agent(
        role="Leo Concept Explainer",
        goal=(
            "Teach the requested topic clearly and accurately according to "
            "the Coordinator's learning plan."
        ),
        backstory=(
            "You are Leo's teaching specialist. You adapt lessons to the student's "
            "level, explain progressively, and use examples and analogies."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )

    quiz_master = Agent(
        role="Leo Quiz Master",
        goal=(
            "Create high-quality practice questions based only on what the student "
            "has just been taught."
        ),
        backstory=(
            "You are Leo's assessment-design specialist. You create fair, educational "
            "multiple-choice questions based on the Coordinator's plan and the "
            "Explainer's lesson."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )

    evaluator = Agent(
        role="Leo Student Answer Evaluator",
        goal=(
            "Evaluate the student's quiz answers accurately, identify strengths "
            "and weaknesses, and provide constructive feedback."
        ),
        backstory=(
            "You are Leo's feedback specialist. You evaluate the exact quiz created "
            "by the Quiz Master, calculate the score, identify misconceptions, and "
            "recommend what the student should revise next."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=5,
    )

    return coordinator, explainer, quiz_master, evaluator


def build_teaching_crew(llm: LLM):
    coordinator, explainer, quiz_master, _ = build_agents(llm)

    planning_task = Task(
        description="""
Student Name: {student_name}
Topic: {topic}
Learning Level: {level}
Previous Performance: {previous_score}

Analyze the student's request and create a concise teaching plan with:
1. Learning goal
2. 4-6 key concepts
3. Recommended teaching sequence
4. Level-appropriate difficulty
5. Important prerequisite(s)
6. Guidance for the Explainer
7. Guidance for the Quiz Master

If the topic is broad or unclear, narrow it to a reasonable introductory lesson.
""",
        expected_output=(
            "A clear teaching plan containing the learning goal, key concepts, "
            "teaching sequence, prerequisites, and guidance for the other agents."
        ),
        agent=coordinator,
    )

    explanation_task = Task(
        description="""
Teach the student the requested topic.

Student: {student_name}
Topic: {topic}
Level: {level}

Follow the Coordinator's learning plan supplied through task context.

Create a lesson with:
1. Simple introduction
2. Key concepts
3. Step-by-step explanation
4. At least two examples
5. One useful analogy where appropriate
6. Common mistakes
7. Short summary
""",
        expected_output=(
            "A clear, student-friendly lesson with explanations, examples, "
            "common mistakes, and a summary."
        ),
        agent=explainer,
        context=[planning_task],
        markdown=True,
    )

    quiz_task = Task(
        description="""
Create a quiz for:

Student: {student_name}
Topic: {topic}
Level: {level}

Use BOTH the Coordinator's plan and the Explainer's lesson supplied through context.

Create EXACTLY 5 multiple-choice questions.

Rules:
- Every question must have exactly 4 options.
- Options correspond to A, B, C, D.
- correct_answer contains only A, B, C, or D.
- Questions test only material that was taught.
- Include a short explanation for every correct answer.
""",
        expected_output=(
            "A structured QuizPack containing the topic and exactly five questions."
        ),
        agent=quiz_master,
        context=[planning_task, explanation_task],
        output_pydantic=QuizPack,
    )

    crew = Crew(
        agents=[coordinator, explainer, quiz_master],
        tasks=[planning_task, explanation_task, quiz_task],
        process=Process.sequential,
        verbose=True,
        memory=True,
    )

    return crew, planning_task, explanation_task, quiz_task


def build_evaluation_crew(llm: LLM, quiz_json: str, student_answers_json: str):
    coordinator, _, _, evaluator = build_agents(llm)

    evaluation_task = Task(
        description=f"""
Evaluate the student's answers.

Student: {{student_name}}
Topic: {{topic}}
Level: {{level}}

QUIZ CREATED BY QUIZ MASTER:
{quiz_json}

STUDENT ANSWERS:
{student_answers_json}

Compare each answer with the correct_answer field.

Return:
1. Number correct
2. Total questions
3. Percentage
4. Correct question numbers
5. Incorrect question numbers
6. Strengths
7. Weaknesses
8. Constructive feedback
9. Recommended next learning action

Do not invent questions and do not change the quiz.
""",
        expected_output="A validated EvaluationResult.",
        agent=evaluator,
        output_pydantic=EvaluationResult,
    )

    final_guidance_task = Task(
        description="""
Review the Evaluator's assessment for:

Student: {student_name}
Topic: {topic}

Using only the Evaluator's result:
1. Congratulate the student appropriately.
2. Summarize current understanding.
3. Identify the most important area to review.
4. Recommend one concrete next learning step.
5. Keep the advice concise and encouraging.

Do not change the Evaluator's score.
""",
        expected_output="Concise final learning guidance from Leo's Coordinator.",
        agent=coordinator,
        context=[evaluation_task],
        markdown=True,
    )

    crew = Crew(
        agents=[evaluator, coordinator],
        tasks=[evaluation_task, final_guidance_task],
        process=Process.sequential,
        verbose=True,
        memory=True,
    )

    return crew, evaluation_task, final_guidance_task


def validate_student_request(name: str, topic: str, level: str) -> list[str]:
    errors = []

    if not name or not name.strip():
        errors.append("Please provide the student's name.")

    if not topic or len(topic.strip()) < 2:
        errors.append("Please provide a valid learning topic.")

    if level not in {"Beginner", "Intermediate", "Advanced"}:
        errors.append("Learning level must be Beginner, Intermediate, or Advanced.")

    return errors


def parse_answers(answer_text: str) -> dict[str, str]:
    answers = [x.strip().upper() for x in answer_text.split(",")]

    if len(answers) != 5:
        raise ValueError(
            "Please enter exactly 5 answers, for example: A,B,C,D,A"
        )

    if any(answer not in {"A", "B", "C", "D"} for answer in answers):
        raise ValueError("Every answer must be A, B, C, or D.")

    return {str(i + 1): answer for i, answer in enumerate(answers)}


def quiz_to_markdown(quiz: QuizPack) -> str:
    text = f"# 🦁 Leo Quiz: {quiz.topic}\n\n"
    letters = ["A", "B", "C", "D"]

    for q in quiz.questions:
        text += f"### Question {q.question_number}\n{q.question}\n\n"
        for letter, option in zip(letters, q.options):
            text += f"**{letter}.** {option}\n\n"

    text += (
        "---\n"
        "### How to answer\n"
        "Enter five answers separated by commas, for example: `A,B,C,D,A`"
    )
    return text
