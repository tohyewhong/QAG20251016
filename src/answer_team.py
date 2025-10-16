from langchain_core.messages import HumanMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agent_utils import ResponderWithRetries
from memory import AnswerGeneration, AnswerRevision, OverallState


def check_answer_explanation(answer, explanation):
    targets = [
        "user's question",
        "user's answer",
        "user's explanation",
        "original question",
        "original answer",
        "original explanation",
        "upon reflection",
        "initial answer",
        "initial explanation",
        "initial question"
    ]
    answer = answer.lower()
    explanation = explanation.lower()
    for target in targets:
        if target in answer or target in explanation:
            return False
    return True


class AnswerSupervisor:
    def __init__(self, model):
        self.model = model
        self.name = "answer_supervisor"
        self.workers = []

        # Init nodes
        self.answer_generator = AnswerGenerator(self.model)
        self.answer_generator_graph = self.answer_generator.create_graph()
        self.workers.append(self.answer_generator.name)

    def create_graph(self):
        builder = StateGraph(OverallState)
        builder.add_node("supervisor", self.supervisor_node)
        builder.add_node(self.answer_generator.name, self.call_answer_generator)
        builder.add_edge(START, "supervisor")
        return builder.compile()

    def supervisor_node(self, state):
        if len(state["questions"]) == len(state["answers"]):
            goto = END
        else:
            goto = self.answer_generator.name
        return Command(goto=goto, update={"next": goto})

    def call_answer_generator(self, state):
        try:
            question = state["questions"][len(state["answers"])]
        except Exception:
            return Command(
                update={
                    "questions": ("add", state["questions"])
                },
                goto="supervisor",
            )

        state["messages"] = [state["messages"][0]]
        state["messages"].append(HumanMessage(
            content=f"### Question:\n\n{question}"
        ))
        response = self.answer_generator_graph.invoke({"messages": state["messages"]})
        answer_to_add = None
        try:
            if len(response["messages"]) == 2:
                # no reflection at all
                message = response["messages"][-1]
            if response["messages"][-1].tool_calls[0]["name"] == "QuestionRevision":
                message = response["messages"][-1]
            else:
                message = response["messages"][-2]

            assert hasattr(message, "tool_calls"), response

            answer = message.tool_calls[0]["args"]["answer"]
            explanation = message.tool_calls[0]["args"]["explanation"]
            if check_answer_explanation(answer, explanation):
                answer_message = (
                    "[Generated the answer and explanation]\n\n"
                    f"Answer: {answer}\n\nExplanation: {explanation}"
                )
                answer_to_add = {
                    "answer": answer,
                    "explanation": explanation
                }
            else:
                print("\nHIT TARGET. RETRYING...\n")
                answer_message = (
                    "Answer and explanation not generated. PLEASE TRY AGAIN."
                )
        except Exception:
            answer_message = (
                "Answer and explanation not generated. PLEASE TRY AGAIN."
            )

        if answer_to_add:
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content=answer_message,
                            name=self.answer_generator.name
                        )
                    ],
                    "questions": ("add", state["questions"]),
                    "answers": [answer_to_add]
                },
                goto="supervisor",
            )
        else:
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content=answer_message,
                            name=self.answer_generator.name
                        )
                    ],
                    "questions": ("add", state["questions"])
                },
                goto="supervisor",
            )


class AnswerGenerator:
    def __init__(self, model):
        self.model = model
        self.name = "answer_generator"
        self.workers = ["generate", "reflect"]

    def create_graph(self):
        builder = StateGraph(OverallState)
        builder.add_node(
            "generate", self.generation_node().respond)
        builder.add_node(
            "reflect", self.reflection_node().respond)
        builder.add_edge(START, "generate")
        builder.add_edge("reflect", "generate")
        builder.add_conditional_edges("generate", self._should_continue)
        return builder.compile()

    def _should_continue(self, state):
        try:
            if "FINISH" in \
                    state["messages"][-1].tool_calls[0]["args"]["recommendation"]:
                return END
        except Exception:
            pass

        if len(state["messages"]) > 6:  # Reduced from 10
            return END
        else:
            return "reflect"

    def get_prompt(self):
        # IMPROVED PROMPT (2025-10-10): Clearer and more concise to reduce failures
        # OLD PROMPT: Was overly verbose with repetitive instructions
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", (
                    "Answer questions using ONLY the provided context.\n"
                    "Requirements: SHORT answer, CLEAR explanation, grounded in context.\n\n"
                    "{first_instruction}\n\n"
                    "Reflect: Is it accurate? Recommend improvements or FINISH if good."
                )),
                MessagesPlaceholder(variable_name="messages"),
                ("system", (
                    "You MUST use the {function_name} function to respond."
                )),
            ]
        )
        return prompt
        
        # OLD PROMPT PRESERVED FOR REFERENCE:
        # prompt = ChatPromptTemplate.from_messages(
        #     [
        #         ("system", (
        #             "You are an expert assistant that answers complex questions"
        #             " based on the given contexts. The answer and explanation"
        #             " must be grounded based on the contexts and does not use"
        #             " any external knowledge.\n\n"
        #             "1. {first_instruction}\n"
        #             "2. Reflect and critique the answer and explanation based on"
        #             " the question's complexity and be specific to the context."
        #             " Be severe to maximize improvement.\n"
        #             "3. Recommend how to improve the answer and explanation,"
        #             " especially if there are critiques of the generated answer"
        #             " and explanation. You MUST ensure that the answer and the"
        #             " explanation are faithful and as natural as possible.\n\n"
        #             "If there is no need to revise the answer and explanation,"
        #             " give your final answer and complete explanation, and respond with"
        #             " FINISH in your recommendation."
        #         )),
        #         MessagesPlaceholder(variable_name="messages"),
        #         ("system", (
        #             "Reflect on the user's original question and the"
        #             " actions taken thus far. You MUST respond using the"
        #             " {function_name} function."
        #         )),
        #     ]
        # )

    def generation_chain(self):
        generate_instruction = (
            "Generate a short answer and clear explanation based on the context."
            " Keep answer concise, explanation complete and natural."
        )
        generate_prompt = self.get_prompt().partial(
            first_instruction=generate_instruction,
            function_name=AnswerGeneration.__name__
        )

        generation_chain = (
            generate_prompt | self.model.bind_tools(tools=[AnswerGeneration]))
        validator = PydanticToolsParser(tools=[AnswerGeneration])
        return generation_chain, validator

    def reflection_chain(self):
        reflect_instruction = (
            "Revise the answer and explanation using the critique."
            " Only improve what needs fixing."
        )
        refelct_prompt = self.get_prompt().partial(
            first_instruction=reflect_instruction,
            function_name=AnswerRevision.__name__
        )

        reflection_chain = (
            refelct_prompt | self.model.bind_tools(tools=[AnswerRevision]))
        validator = PydanticToolsParser(tools=[AnswerRevision])
        return reflection_chain, validator

    def generation_node(self):
        generation_chain, validator = self.generation_chain()
        return ResponderWithRetries(
            runnable=generation_chain, validator=validator
        )

    def reflection_node(self):
        reflection_chain, validator = self.reflection_chain()
        return ResponderWithRetries(
            runnable=reflection_chain, validator=validator
        )
