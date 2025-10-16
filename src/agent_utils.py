from typing import Literal
import time
import random

from langchain_core.messages import ToolMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END
from langgraph.types import Command
from pydantic import ValidationError
from typing_extensions import TypedDict
from openai import InternalServerError, APIConnectionError, APITimeoutError, RateLimitError

from memory import OverallState


def supervisor_system_prompt(workers, goal: str):
    system_prompt = (
        "You are a supervisor tasked with managing a conversation between the"
        f" following workers: {workers}. {goal}\n"
        "Given the following user request, respond with the worker to act next."
        " Each worker will perform a task and respond with their results"
        " and status. When finished, respond with FINISH."
    )
    return system_prompt


def create_supervisor_node(model, system_prompt: str, workers: list[str]):
    # Define list of options for next worker
    options = ["FINISH", *workers]

    # Define response model for worker selection
    class Router(TypedDict):
        next: Literal[*options]  # type: ignore

    def supervisor_node(state: OverallState) -> Command[
            Literal[*workers, "supervisor"]]:  # type: ignore
        """An LLM-based router"""
        supervisor_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            ("system", (
                f"You MUST respond using the {Router.__name__} function."
                " Do NOT call any other functions or tools. Only use the Router"
                " function to select the next worker from the available options:"
                f" {options}"
            )),
        ])

        supervisor_chain = supervisor_prompt | model.bind_tools(
            tools=[Router],
            tool_choice="Router"
        )
        validator = PydanticToolsParser(tools=[Router])

        goto = ""
        if len(state["questions"]) >= 10 or len(state["messages"]) >= 20:
            goto = END
        else:
            for attempt in range(2):  # Reduced from 3
                try:
                    response = ResponderWithRetries(
                        runnable=supervisor_chain, validator=validator
                    ).respond(state)
                    # response is a dict with "messages" key containing a single AIMessage
                    msg = response.get("messages") if isinstance(response, dict) else response
                    # Handle both single message and list of messages
                    if isinstance(msg, list):
                        # If it's a list, find the last one with tool_calls
                        for m in reversed(msg):
                            if hasattr(m, "tool_calls") and m.tool_calls:
                                msg = m
                                break
                    
                    # Now msg should be a single AIMessage
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        # Check if the tool call is for Router
                        tool_call = msg.tool_calls[0]
                        if tool_call.get("name") == "Router":
                            goto = tool_call["args"]["next"]  # type: ignore
                            break
                except (KeyError, AttributeError, IndexError, TypeError) as e:
                    print(f"\n[Supervisor] Attempt {attempt + 1}/2 failed with error: {e}")
                    if attempt == 1:  # Adjusted for reduced attempts
                        # Fallback: END the graph
                        print("[Supervisor] Max attempts reached, ending graph")
                        goto = END
            if goto == "FINISH" or goto == "":
                goto = END

        return Command(goto=goto, update={"next": goto})

    return supervisor_node


def get_next_node(node):
    return node["next"]


class ResponderWithRetries:
    def __init__(self, runnable, validator):
        self.runnable = runnable
        self.validator = validator

    def respond(self, state):
        response = []
        max_validation_attempts = 2  # Reduced from 5
        max_request_retries = 2      # Reduced from 5
        
        for attempt in range(max_validation_attempts):
            # Retry with exponential backoff for transient API errors
            for retry in range(max_request_retries):
                try:
                    response = self.runnable.invoke(
                        {"messages": state["messages"]},
                        {
                            "tags": [f"attempt:{attempt}", f"retry:{retry}"],
                            "recursion_limit": 35
                        }
                    )
                    # Successfully got response, break out of retry loop
                    break
                except (InternalServerError, APIConnectionError, APITimeoutError, RateLimitError) as e:
                    if retry < max_request_retries - 1:
                        # Exponential backoff with jitter: 2^retry * base_delay + random jitter
                        base_delay = 2.0
                        max_jitter = 1.0
                        wait_time = (2 ** retry) * base_delay + random.uniform(0, max_jitter)
                        print(f"\n[ResponderWithRetries] Transient API error ({type(e).__name__}): {e}")
                        print(f"[ResponderWithRetries] Retrying in {wait_time:.2f}s (retry {retry + 1}/{max_request_retries})...\n")
                        time.sleep(wait_time)
                    else:
                        # Max retries exceeded, re-raise the error
                        print(f"\n[ResponderWithRetries] Max retries ({max_request_retries}) exceeded. Failing.\n")
                        raise
            
            # Now validate the response
            try:
                self.validator.invoke(response)
                return {"messages": response}
            except ValidationError as e:
                state["messages"].append(response)
                state["messages"].append(ToolMessage(
                    content=(
                        f"{e!r}\n\nPay close attention to the"
                        " function schema.\n\n"
                        f"{self.validator.schema_json()}"
                        " Respond by fixing all validation errors."
                    ),
                    tool_call_id=response.tool_calls[0]["id"],
                ))

        return {"messages": response}
