import operator
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# For Question Team
# Supervisor


def update_questions(
    existing: list[str],
    update: tuple[str, list[str]]
):
    if existing and update[0] == "remove":
        existing = [qn for qn in existing if qn not in update[1]]
        return existing
    elif not update:
        return (existing or []) + update

    to_update = [qn for qn in update[1] if qn not in existing]
    return (existing or []) + to_update


class OverallState(TypedDict):
    messages: Annotated[list, add_messages]
    questions: Annotated[list[str], update_questions]
    answers: Annotated[list[dict], operator.add]


# Generate questions (first and subsequent)
class QuestionGeneration(BaseModel):
    """Based on the given context, generate a complex
    question and then reflect on the question generated.

    The question MUST BE complex which means that it requires
    multiple reasoning steps to answer it. The question MUST BE
    answerable only based on information from the context and
    does not require any additional information.
    """
    question: str = Field(
        description=(
            "A generated complex question based on the context."
            " The question MUST BE complex which means that it requires"
            " multiple reasoning steps to answer it. The question"
            " MUST BE answerable only based on information from the"
            " context and does not require any additional information."
            " The question MUST NOT have any reasoning shortcuts that"
            " makes allows someone to answer the question easily."
        ))
    reflection: str = Field(
        description=(
            "Your reflection of the generated complex question."
            " Be severe to maximize improvement."
        ))
    recommendation: str = Field(
        description=(
            " Your recommendation of how to improve the generated"
            " complex question. Be has detailed as possible to"
            " maximize improvement."
        ))


class QuestionRevision(QuestionGeneration):
    """Revise your original question based on the context and
    the new information given. You should use the previous critique
    to improve the question. Then, reflect on the generated
    question again.
    """
    pass


# Check questions
class CheckQuestionsGeneration(BaseModel):
    """Based on the given context and list of complex
    questions, decide which questions are unnecessary and should
    be removed from the list.
    """
    questions_to_remove: list[str] = Field(
        description=(
            "The list of questions to remove from the current list of"
            " questions based on the given context. Your goal is to ensure"
            " that the list of questions are of diverse question types"
            " and cover as much of the context as possible."
        ))
    reflection: str = Field(
        description=(
            "Your reflection of selection of the list of questions to remove."
            " Be severe to maximize improvement."
        ))
    recommendation: str = Field(
        description=(
            " Your recommendation of how to improve the improve the current list"
            " of questions and whether questions should be kept or removed."
            " Be has detailed as possible to maximize improvement."
        ))


class CheckQuestionsRevision(CheckQuestionsGeneration):
    """Revise your original question based on the context and
    the new information given. You should use the previous critique
    to improve the question. Then, reflect on the generated
    question again.
    """
    pass


# Answer
class AnswerGeneration(BaseModel):
    """Based on the given context and complex
    question, generate the answer and explanation
    and then reflect on the question generated.

    The answer and explanation MUST answer the question fully.
    The answer and explanation MUST ONLY use information from
    the context and MUST NOT use any external information.
    """
    answer: str = Field(
        description=(
            "A short and concise answer based on the context and question."
            " The answer must answer the question fully and not use any external"
            " information outside of the provided contexts. The answer has"
            " to be precise, clear, and as natural as possible."
        ))
    explanation: str = Field(
        description=(
            "A explanation about the answer based on the context and question."
            " The explanation can be a long-form answer that explains the reasoning"
            " and thinking steps needed to arrive at the final answer."
            " The explanation must answer the question fully and not use any external"
            " information outside of the provided contexts. The explanation has"
            " to be clear and as natural as possible."
        ))
    reflection: str = Field(
        description=(
            "Your reflection of the generated answer and explanation."
            " Be severe to maximize improvement."
        ))
    recommendation: str = Field(
        description=(
            " Your recommendation of how to improve the generated"
            " answer and explanation. Be as detailed as possible to"
            " maximize improvement."
        ))


class AnswerRevision(AnswerGeneration):
    """"Revise and improve your answer and explanation"
    using the critique and recommendations. You MUST only REVISE the
    answer and explanation. Then, reflect on the generated question again.
    """
    pass
