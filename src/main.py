# This is a Question-Answer Generation (QAG) Agent that uses LangGraph to create a multi-agent system 
# for generating complex questions and answers from given text contexts. The system follows a two-stage process: 
# first generating questions, then answering them.
import argparse
import json
import os

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from tqdm import tqdm

from answer_team import AnswerSupervisor
from memory import OverallState
from question_team import QuestionSupervisor
from utils import Config, load_yaml_config

# 1. Configuration Management

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="default.yaml")
    args = parser.parse_args()

    config_path = os.path.join(os.getcwd(), "configs", args.config)
    config = Config(**load_yaml_config(config_path))
    return config


class QAGAgent:
    def __init__(self, model):
        self.model = model
        self.name = "qag_agent"

        # Init nodes
        self.question_supervisor = QuestionSupervisor(self.model)
        self.question_supervisor_graph = self.question_supervisor.create_graph()

        self.answer_supervisor = AnswerSupervisor(self.model)
        self.answer_supervisor_graph = self.answer_supervisor.create_graph()

    def create_graph(self):
        graph = StateGraph(OverallState)
        graph.add_node(self.question_supervisor.name, self.call_question_supervisor)
        graph.add_node(self.answer_supervisor.name, self.call_answer_supervisor)
        graph.add_edge(START, self.question_supervisor.name)
        graph.add_edge(self.question_supervisor.name, self.answer_supervisor.name)
        graph.add_edge(self.answer_supervisor.name, END)
        return graph.compile()

    def call_question_supervisor(self, state):
        response = self.question_supervisor_graph.invoke({
            "messages": state["messages"]})

        print("\n\nQuestions:", response["questions"])
        return {
            "messages": response["messages"][0],
            "questions": ("add", response["questions"])
        }

    def call_answer_supervisor(self, state):
        response = self.answer_supervisor_graph.invoke({
            "messages": state["messages"],
            "questions": ("add", state["questions"])})

        print("\n\nAnswers:", response["answers"])
        return {
            "messages": response["messages"][0],
            "questions": ("add", response["questions"]),
            "answers": response["answers"]
        }


def run_graph(graph, example):
    results = graph.invoke(
        {"messages": f"### Context:\n\n{example.strip()}"},
        {"recursion_limit": 50}
    )
    questions = results["questions"]
    answers = results["answers"]
    return questions, answers


def get_data(config):
    def prepare_example(text):
        if isinstance(text, str):
            return text
        else:
            prepared_text = ""
            for i in range(len(text)):
                prepared_text += f"Document {i + 1}:\n{text[i]}\n\n"
            prepared_text.strip()
            return prepared_text

    # Data file needs to be a jsonl file, with "text" key and a list of strings
    with open(config.data_path, "r") as f:
        lines = [json.loads(line) for line in f.readlines()]
    return [prepare_example(line["text"]) for line in lines]


def main():
    config = get_config()
    kwargs = {
        "model_name": config.model_name,
        "openai_api_key": config.openai_api_key,
        "openai_api_base": config.openai_api_base,
        "temperature": 0.8,
        "max_retries": 3,
        "timeout": 120.0,
        "request_timeout": 120.0,
    }
    model = ChatOpenAI(**kwargs)  # type: ignore

    data = get_data(config)
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir, exist_ok=True)

    qag_agent = QAGAgent(model)
    graph = qag_agent.create_graph()

    for i, example in tqdm(enumerate(data), total=len(data)):
        output_path = os.path.join(config.output_dir, f"output_{i}.json")
        if os.path.exists(output_path):
            print(f"{output_path} exists!\n")
            continue
        questions, answers = run_graph(graph, example)
        with open(output_path, "w") as f:
            json.dump({
                "context": example,
                "questions": questions,
                "answers": answers
            }, f, indent=4)


if __name__ == "__main__":
    main()
