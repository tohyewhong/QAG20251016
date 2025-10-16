import logging
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agent_utils import ResponderWithRetries, create_supervisor_node
from memory import (
    CheckQuestionsGeneration,
    CheckQuestionsRevision,
    OverallState,
    QuestionGeneration,
    QuestionRevision,
)

# Configure logging for question generation
logger = logging.getLogger(__name__)

# Configure file handler for logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Create a unique log file name with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"question_generation_{timestamp}.log")

# Configure file handler
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Configure formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add file handler to logger
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

# Also log to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class QuestionSupervisor:
    def __init__(self, model):
        self.model = model
        self.name = "question_supervisor"
        self.workers = []

        # Init nodes
        self.first_generator = FirstQuestionGenerator(self.model)
        self.first_generator_graph = self.first_generator.create_graph()
        self.workers.append(self.first_generator.name)

        self.next_generator = NextQuestionGenerator(self.model)
        self.next_generator_graph = self.next_generator.create_graph()
        self.workers.append(self.next_generator.name)

        self.check_questions = CheckQuestions(self.model)
        self.check_questions_graph = self.check_questions.create_graph()
        self.workers.append(self.check_questions.name)

        self.supervisor_node = create_supervisor_node(
            model,
            (
                "You are a supervisor tasked with managing a conversation between the"
                f" following workers: {self.workers}.\n\n"
                f"The {self.first_generator.name} generates the first question based"
                " on the context.\n"
                f"The {self.next_generator.name} generates subsequent questions based"
                " on the provided context.\n"
                f"The {self.check_questions.name} decides which questions to remove"
                " from the current list of questions --"
                " when questions are not complete, duplicated, or not diverse enough."
                " You should check the questions whenever you deem necessary but"
                " there should not be sequential checking of questions. A check should"
                " be followed by generating more questions or FINISH.\n\n"
                "Your goal is to generate a diverse list of complex questions based"
                " on the given context. Remember that when you have multiple questions,"
                " you MUST check that the questions cover as much information in the"
                " context as possible and are diverse. The questions MUST BE answerable"
                " based on only using information from the contexts and does"
                " not require any external information and knowledge."
                " The questiosn MUST BE as specific as possible."
                " A complex question is one that requires"
                " multiple reasoning steps before arriving at the answer."
                " The questions MUST NOT have any reasoning shortcuts that"
                " makes allows someone to answer the question easily.\n\n"
                "Given the following user request, respond with the worker to act next."
                " Each worker will perform a task and respond with their results"
                " and status. When finished, respond with FINISH."
            ),
            self.workers
        )

    def create_graph(self):
        logger.info(f"Creating graph for {self.name}")
        builder = StateGraph(OverallState)
        builder.add_node("supervisor", self.supervisor_node)
        builder.add_node(self.first_generator.name, self.call_first_generator)
        builder.add_node(self.next_generator.name, self.call_next_generator)
        builder.add_node(self.check_questions.name, self.call_check_questions)
        builder.add_edge(START, "supervisor")
        logger.info(f"Graph created successfully for {self.name}")
        return builder.compile()

    def call_first_generator(self, state):
        # Log current state before generation
        logger.info(f"Current questions in list: {len(state['questions'])}")
        
        response = self.first_generator_graph.invoke({
            "messages": state["messages"],
            "questions": ("add", state["questions"])})

        try:
            if len(response["messages"]) == 2:
                # no reflection at all
                message = response["messages"][-1]
            if response["messages"][-1].tool_calls[0]["name"] == "QuestionRevision":
                message = response["messages"][-1]
            else:
                message = response["messages"][-2]

            assert hasattr(message, "tool_calls"), response
            question = message.tool_calls[0]["args"]["question"]
            
            # Log the generated question
            logger.info(f"First question generated: {question}")
            
            question_message = (
                "[Generated the first question]\n\n"
                f"Here is the first question: {question}"
            )
            question_to_add = [question]
        except Exception as e:
            # Log failed question generation
            logger.warning(f"First question generation failed: {str(e)}")
            question_message = (
                "First question not generated. PLEASE TRY AGAIN."
            )
            question_to_add = []
        
        # Log the number of questions being added
        logger.info(f"Adding {len(question_to_add)} first question(s) to the list")

        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=question_message,
                        name=self.first_generator.name
                    )
                ],
                "questions": ("add", question_to_add)
            },
            goto="supervisor",
        )

    def call_next_generator(self, state):
        # Log current state before generation
        logger.info(f"Current questions in list: {len(state['questions'])}")
        
        current_list = (
            "Generate the next question.\n"
            "Here is the current list of questions:\n"
        )
        for question in state["questions"]:
            current_list += f"- {question}\n"
        current_list = current_list.strip()

        state["messages"].append(HumanMessage(content=current_list))
        prev_length = len(state["messages"])
        response = self.next_generator_graph.invoke({
            "messages": state["messages"],
            "questions": ("add", state["questions"])})

        curr_length = len(response["messages"])
        try:
            if curr_length - prev_length == 1:
                message = response["messages"][-1]
            if response["messages"][-1].tool_calls[0]["name"] == "QuestionRevision":
                message = response["messages"][-1]
            else:
                message = response["messages"][-2]

            assert hasattr(message, "tool_calls"), response
            question = message.tool_calls[0]["args"]["question"]
            
            # Log the generated question
            logger.info(f"Next question generated: {question}")
            
            question_message = (
                "[Generated the next question]\n\n"
                f"Here is the next question: {question}"
            )
            question_to_add = [question]
        except Exception as e:
            # Log failed question generation
            logger.warning(f"Next question generation failed: {str(e)}")
            question_message = "Next question not generated. PLEASE TRY AGAIN."
            question_to_add = []
        
        # Log the number of questions being added
        logger.info(f"Adding {len(question_to_add)} next question(s) to the list")

        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=question_message,
                        name=self.next_generator.name
                    )
                ],
                "questions": ("add", question_to_add)
            },
            goto="supervisor",
        )

    def call_check_questions(self, state):
        # Log current state before checking
        logger.info(f"Checking questions list with {len(state['questions'])} question(s)")
        
        current_list = (
            "Check the current list of questions.\n"
            "Here is the current list of questions:\n"
        )
        for question in state["questions"]:
            current_list += f"- {question}\n"
        current_list = current_list.strip()

        state["messages"].append(HumanMessage(content=current_list))

        questions_to_remove = []
        recommendation = ""

        prev_length = len(state["messages"])
        response = self.check_questions_graph.invoke({
            "messages": state["messages"],
            "questions": ("add", state["questions"])})
        curr_length = len(response["messages"])

        try:
            if curr_length - prev_length == 1:
                message = response["messages"][-1]
            if response["messages"][-1].tool_calls[0]["name"] == \
                    "CheckQuestionsRevision":
                message = response["messages"][-1]
            else:
                message = response["messages"][-2]

            assert hasattr(message, "tool_calls"), response
            questions_to_remove = message.tool_calls[0]["args"]["questions_to_remove"]
            if isinstance(questions_to_remove, list) and len(questions_to_remove) > 0:
                # Log questions being removed
                logger.info(f"Questions being removed by supervisor: {questions_to_remove}")
                logger.info(f"Removing {len(questions_to_remove)} question(s) from the list")
                removed_list = "Here are the questions removed:\n"
                for question in questions_to_remove:
                    removed_list += f"- {question}\n"
                removed_list = removed_list.strip()
            else:
                logger.info("No questions removed by supervisor")
                logger.info("Removing 0 question(s) from the list")
                removed_list = "There is no need to edit the current list of questions."

            reflection = message.tool_calls[0]["args"]["reflection"]
            recommendation = message.tool_calls[0]["args"]["recommendation"]
            if recommendation == "FINISH":
                recommendation = ""

            check_message = f"[Checked questions]\n\n{removed_list}"
            check_message += f"\n\nReflection: {reflection}"
            if len(recommendation) > 0:
                check_message += f"\n\nRecommendation: {recommendation}"

        except Exception as e:
            logger.warning(f"Question checking failed: {str(e)}")
            check_message = "There is no need to edit the current list of questions."

        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=check_message,
                        name=self.check_questions.name
                    )
                ],
                "questions": ("remove", questions_to_remove)
            },
            goto="supervisor",
        )


class FirstQuestionGenerator:
    def __init__(self, model):
        self.model = model
        self.name = "first_question_generator"
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

        if len(state["messages"]) > 10:
            return END
        else:
            return "reflect"

    def get_prompt(self):
        # IMPROVED PROMPT (2025-10-10): Simplified and more direct to reduce failures and speed up generation
        # OLD PROMPT: Was too verbose and had grammatical errors ("makes allows someone"), causing confusion
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", (
                    "You are an expert at creating complex questions from given contexts.\n\n"
                    "REQUIREMENTS:\n"
                    "- Question must be answerable using ONLY the provided contexts\n"
                    "- Question must require multiple reasoning steps\n"
                    "- Question must be specific and detailed (max 30 words)\n"
                    "- For multiple documents, question should require cross-document reasoning\n"
                    "- Avoid questions with obvious shortcuts\n\n"
                    "{first_instruction}\n\n"
                    "Then reflect: Is this question complex enough and answerable from the context?\n"
                    "Recommend improvements if needed, or respond with FINISH if the question is good."
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
        #             "You are an expert assistant that creates a complex question"
        #             " based on the given contexts.\n\nThe question MUST BE answerable"
        #             " based on only using information from the contexts and does"
        #             " not require any external information and knowledge."
        #             " The question MUST BE as specific as possible."
        #             " A complex question is one that requires"
        #             " multiple reasoning steps before arriving at the answer."
        #             " If there are multiple documents in the context, then the"
        #             " question should require information across the various"
        #             " contexts before it can be answered. The question MUST NOT"
        #             " have any reasoning shortcuts that makes allows someone"
        #             " to answer the question easily.\n\n"
        #             "1. {first_instruction}\n"
        #             "2. Reflect and critique the generated question based on"
        #             " its complexity and specific to the context."
        #             " Be severe to maximize improvement.\n"
        #             "3. Recommend how to improve the question, especially if"
        #             " there are critiques of the generated question."
        #             " Make sure that the generated question is"
        #             " answerable and as natural as possible.\n\n"
        #             "If there is no need to revise the question,"
        #             " respond with FINISH in your recommendation."
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
            "Generate a complex question based on the provided contexts."
            " The question MUST BE fewer than 30 words."
        )
        generate_prompt = self.get_prompt().partial(
            first_instruction=generate_instruction,
            function_name=QuestionGeneration.__name__
        )

        generation_chain = (
            generate_prompt | self.model.bind_tools(tools=[QuestionGeneration]))
        validator = PydanticToolsParser(tools=[QuestionGeneration])
        return generation_chain, validator

    def reflection_chain(self):
        reflect_instruction = (
            "Revise and improve your generated question"
            " using the critique and recommendations."
            " You MUST NOT generate a new question and"
            " only REVISE the given generated question."
        )
        refelct_prompt = self.get_prompt().partial(
            first_instruction=reflect_instruction,
            function_name=QuestionRevision.__name__
        )

        reflection_chain = (
            refelct_prompt | self.model.bind_tools(tools=[QuestionRevision]))
        validator = PydanticToolsParser(tools=[QuestionRevision])
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


class NextQuestionGenerator(FirstQuestionGenerator):
    def __init__(self, model):
        self.model = model
        self.name = "next_question_generator"
        self.workers = ["generate", "reflect"]

    def get_prompt(self):
        # IMPROVED PROMPT (2025-10-10): Clearer structure, fixed grammar errors
        # OLD PROMPT: Had fragment "If there aredont remove any questions" which confused the model
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", (
                    "You create NEW complex questions that are DIFFERENT from existing questions.\n\n"
                    "REQUIREMENTS:\n"
                    "- Create a question of a DIFFERENT TYPE than those already generated\n"
                    "- Ensure diversity: cover new aspects of the context\n"
                    "- Must be answerable using ONLY the provided contexts\n"
                    "- Requires multiple reasoning steps\n"
                    "- Specific and detailed (max 30 words)\n"
                    "- For multiple documents, require cross-document reasoning\n\n"
                    "{first_instruction}\n\n"
                    "Then reflect: Is this question diverse, complex, and answerable?\n"
                    "Recommend improvements or respond with FINISH if good."
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
        #             "You are an expert assistant that creates a NEW and DIFFERENT"
        #             " complex question based on the given context and current list of"
        #             " complex questions that have already been generated.\n\n"
        #             " The new question MUST BE of a different question type than the"
        #             " questions in the current list. You must ENSURE that the list of"
        #             " is as diverse as possible and cover the context as"
        #             " much as possible. The question MUST BE answerable based on"
        #             " only using information from the contexts and does not require"
        #             " any external information and knowledge. The question MUST BE"
        #             " as specific as possible. A complex question is one that requires"
        #             " multiple reasoning steps before arriving at the answer."
        #             " If there aredont remove any questions. context, then the"
        #             " question should require information across the various"
        #             " contexts before it can be answered. The question MUST NOT"
        #             " have any reasoning shortcuts that makes allows someone"
        #             " to answer the question easily.\n\n"
        #             "1. {first_instruction}\n"
        #             "2. Reflect and critique the generated question based on"
        #             " its complexity, specific to the context, and whether the new"
        #             " question is different and diverse from the current questions."
        #             " Be severe to maximize improvement.\n"
        #             "3. Recommend how to improve the question, especially if"
        #             " there are critiques of the generated question."
        #             " Make sure that the generated question is"
        #             " answerable, as natural as possible, different, and diverse.\n\n"
        #             "If there is no need to revise the question,"
        #             " respond with FINISH in your recommendation."
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
            "Generate a new complex question based on the provided contexts"
            " and current list of questions. The question MUST BE fewer than 30 words."
        )
        generate_prompt = self.get_prompt().partial(
            first_instruction=generate_instruction,
            function_name=QuestionGeneration.__name__
        )

        generation_chain = (
            generate_prompt | self.model.bind_tools(tools=[QuestionGeneration]))
        validator = PydanticToolsParser(tools=[QuestionGeneration])
        return generation_chain, validator


class CheckQuestions(FirstQuestionGenerator):
    def __init__(self, model):
        self.model = model
        self.name = "questions_checker"
        self.workers = ["generate", "reflect"]

    def get_prompt(self):
        # IMPROVED PROMPT (2025-10-10): Simplified - currently set to keep all questions to speed up processing
        # This checker was causing slowdowns; keeping it simple for now
        # OLD PROMPT: Complex validation logic that was time-consuming
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", (
                    "You validate a list of questions generated from a context.\n\n"
                    "Current mode: ACCEPT ALL questions as valid.\n"
                    "Just confirm all questions are acceptable.\n\n"
                    "{first_instruction}\n\n"
                    "Respond with empty questions_to_remove list and FINISH in recommendation."
                )),
                MessagesPlaceholder(variable_name="messages"),
                ("system", (
                    "You MUST use the {function_name} function to respond."
                )),
            ]
        )
        return prompt
        
        # OLD COMPLEX VALIDATION PROMPT (disabled to speed up processing):
        # To re-enable full validation, uncomment below and comment out the simple prompt above
        # prompt = ChatPromptTemplate.from_messages(
        #     [
        #         ("system", (
        #             "You are an expert assistant that checks a list of complex"
        #             " questions generated for a context.\n\n"
        #             "The list of questions MUST BE diverse and of different"
        #             " types. The list of questions MUST cover as much information"
        #             " in the context as possible. The questions in the list MUST BE"
        #             " based only on information from the contexts and does"
        #             " not require any external information and knowledge."
        #             " The question MUST BE as specific as possible."
        #             " A complex question is one that requires"
        #             " multiple reasoning steps before arriving at the answer."
        #             " If there are multiple documents in the context, then the"
        #             " question should require information across the various"
        #             " contexts before it can be answered. The question MUST NOT"
        #             " have any reasoning shortcuts that makes allows someone"
        #             " to answer the question easily.\n\n"
        #             "1. {first_instruction}\n"
        #             "2. Reflect and critique the list of questions you want to remove"
        #             " based on the context, diversity of questions, and"
        #             " question coverage. Be severe to maximize improvement.\n"
        #             "3. Recommend how to improve the list of the questions,"
        #             " especially if there are critiques of your list of questions"
        #             " to remove.\n\nMake sure that the questions in your list"
        #             " can be found in the list of initial questions provided."
        #             " If there is no need to edit the list of questions,"
        #             " respond only with FINISH in your recommendation and nothing else."
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
            "Select questions to remove, if any, from the current list of"
            " questions. You can select more than one question to remove."
            " These questions are removed because they do not add to"
            " either the diversity of questions or coverage of the context."
        )
        generate_prompt = self.get_prompt().partial(
            first_instruction=generate_instruction,
            function_name=CheckQuestionsGeneration.__name__
        )

        generation_chain = (
            generate_prompt | self.model.bind_tools(tools=[CheckQuestionsGeneration]))
        validator = PydanticToolsParser(tools=[CheckQuestionsGeneration])
        return generation_chain, validator

    def reflection_chain(self):
        reflect_instruction = (
            "Revise and improve your list of questions to remove"
            " using the critique and recommendations. You should"
            " use the previous critique to improve the list."
            " Then, reflect on the list of questions again."
        )
        refelct_prompt = self.get_prompt().partial(
            first_instruction=reflect_instruction,
            function_name=CheckQuestionsRevision.__name__
        )

        reflection_chain = (
            refelct_prompt | self.model.bind_tools(tools=[CheckQuestionsRevision]))
        validator = PydanticToolsParser(tools=[CheckQuestionsRevision])
        return reflection_chain, validator
