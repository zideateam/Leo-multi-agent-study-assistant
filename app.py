from __future__ import annotations

import json

import gradio as gr

from leo_core import (
    build_evaluation_crew,
    build_llm,
    build_teaching_crew,
    parse_answers,
    quiz_to_markdown,
    validate_student_request,
)
from memory_store import get_student_history, update_student_memory


llm = build_llm()

# Simple single-user demo session state.
current_session = {
    "quiz": None,
    "student_name": None,
    "topic": None,
    "level": None,
}


async def start_learning(student_name: str, topic: str, level: str):
    errors = validate_student_request(student_name, topic, level)
    if errors:
        return "❌ " + "\n\n❌ ".join(errors), "", ""

    try:
        history = get_student_history(student_name)
        previous_score = history.get("previous_score")

        crew, planning_task, explanation_task, quiz_task = build_teaching_crew(llm)

        inputs = {
            "student_name": student_name.strip(),
            "topic": topic.strip(),
            "level": level,
            "previous_score": (
                f"{previous_score:.1f}%"
                if isinstance(previous_score, (int, float))
                else "No previous score"
            ),
        }

        # Colab/Jupyter-safe async execution.
        await crew.kickoff_async(inputs=inputs)

        quiz = quiz_task.output.pydantic
        if quiz is None:
            raise RuntimeError("Quiz Master did not return a valid structured quiz.")

        current_session["quiz"] = quiz
        current_session["student_name"] = student_name.strip()
        current_session["topic"] = topic.strip()
        current_session["level"] = level

        status = """
### 🤖 Agent Activity

✅ **Coordinator** — Learning plan completed  
✅ **Explainer** — Lesson completed  
✅ **Quiz Master** — Structured quiz generated  
⏳ **Evaluator** — Waiting for student answers
"""

        return status, explanation_task.output.raw, quiz_to_markdown(quiz)

    except Exception as exc:
        return (
            "### ⚠️ Coordinator Error Handler\n\n"
            f"Leo could not complete the learning workflow.\n\n`{exc}`",
            "",
            "",
        )


async def evaluate_answers(answer_text: str):
    quiz = current_session.get("quiz")
    if quiz is None:
        return "❌ Please click **Start Learning** first."

    try:
        student_answers = parse_answers(answer_text)

        quiz_json = quiz.model_dump_json(indent=2)
        answers_json = json.dumps(student_answers, indent=2)

        crew, evaluation_task, final_guidance_task = build_evaluation_crew(
            llm,
            quiz_json,
            answers_json,
        )

        inputs = {
            "student_name": current_session["student_name"],
            "topic": current_session["topic"],
            "level": current_session["level"],
        }

        await crew.kickoff_async(inputs=inputs)

        evaluation = evaluation_task.output.pydantic
        if evaluation is None:
            raise RuntimeError("Evaluator did not return a valid structured result.")

        update_student_memory(
            student_name=current_session["student_name"],
            topic=current_session["topic"],
            level=current_session["level"],
            score=evaluation.score,
            total_questions=evaluation.total_questions,
            percentage=evaluation.percentage,
        )

        strengths = "\n".join(f"- ✅ {x}" for x in evaluation.strengths)
        weaknesses = "\n".join(f"- ⚠️ {x}" for x in evaluation.weaknesses)

        return f"""
# 📝 Leo Evaluation

## Score
**{evaluation.score}/{evaluation.total_questions}**  
**{evaluation.percentage:.1f}%**

## Strengths
{strengths}

## Needs Improvement
{weaknesses}

## Evaluator Feedback
{evaluation.feedback}

## Recommended Revision
{evaluation.recommendation}

---

## 🧭 Coordinator's Final Advice
{final_guidance_task.output.raw}

---

### 🤖 Final Agent Status
✅ Coordinator — Completed  
✅ Explainer — Completed  
✅ Quiz Master — Completed  
✅ Evaluator — Completed  
✅ Student memory — Updated
"""

    except Exception as exc:
        return (
            "### ⚠️ Coordinator Error Handler\n\n"
            "Leo could not evaluate the answers.\n\n"
            f"`{exc}`\n\n"
            "Use the format: `A,B,C,D,A`"
        )


with gr.Blocks(title="Leo Multi-Agent Study Assistant") as demo:
    gr.Markdown(
        """
# 🦁 Leo — Multi-Agent Study Assistant

Leo uses multiple AI agents that collaborate to help a student learn.

**Workflow:** Coordinator → Explainer → Quiz Master → Student → Evaluator → Coordinator
"""
    )

    with gr.Row():
        student_name = gr.Textbox(
            label="Student Name",
            placeholder="Example: Zubair",
        )
        topic = gr.Textbox(
            label="Learning Topic",
            placeholder="Example: Python Dictionaries",
        )
        level = gr.Dropdown(
            choices=["Beginner", "Intermediate", "Advanced"],
            value="Beginner",
            label="Learning Level",
        )

    start_button = gr.Button("🚀 Start Learning", variant="primary")

    agent_status = gr.Markdown()
    gr.Markdown("---")
    gr.Markdown("## 📚 Explainer Lesson")
    lesson_output = gr.Markdown()
    gr.Markdown("---")
    quiz_output = gr.Markdown()

    answer_input = gr.Textbox(
        label="Your Answers",
        placeholder="Example: A,B,C,D,A",
    )
    submit_button = gr.Button("📝 Submit Answers")
    evaluation_output = gr.Markdown()

    start_button.click(
        fn=start_learning,
        inputs=[student_name, topic, level],
        outputs=[agent_status, lesson_output, quiz_output],
    )

    submit_button.click(
        fn=evaluate_answers,
        inputs=[answer_input],
        outputs=evaluation_output,
    )


if __name__ == "__main__":
    demo.launch()
