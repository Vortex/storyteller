import faiss
import math
from lib import webuiLLM
from langchain.vectorstores import FAISS
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.docstore import InMemoryDocstore
from langchain.experimental.generative_agents import GenerativeAgent, GenerativeAgentMemory
from termcolor import colored
from typing import List
from datetime import datetime, timedelta
import logging
logging.basicConfig(level=logging.ERROR)

USER_NAME = "Person A"  # The name you want to use when interviewing the agent.
LLM = webuiLLM()  # Can be any LLM you want.


def relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    # This will differ depending on a few things:
    # - the distance / similarity metric used by the VectorStore
    # - the scale of your embeddings (OpenAI's are unit norm. Many others are not!)
    # This function converts the euclidean norm of normalized embeddings
    # (0 is most similar, sqrt(2) most dissimilar)
    # to a similarity function (0 to 1)
    return 1.0 - score / math.sqrt(2)


def create_new_memory_retriever():
    """Create a new vector store retriever unique to the agent."""
    # Define your embedding model
    embeddings_model = SentenceTransformerEmbeddings(
        model_name="flax-sentence-embeddings/all_datasets_v4_MiniLM-L6")

    # Initialize the vectorstore as empty
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore(
        {}), {}, relevance_score_fn=relevance_score_fn)
    return TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, other_score_keys=["importance"], k=15)


tommies_memory = GenerativeAgentMemory(
    llm=LLM,
    memory_retriever=create_new_memory_retriever(),
    verbose=False,
    # we will give this a relatively low number to show how reflection works
    reflection_threshold=8
)

tommie = GenerativeAgent(name="Tommie",
                         age=25,
                         # You can add more persistent traits here
                         traits="anxious, likes design, talkative",
                         # When connected to a virtual world, we can have the characters update their status
                         status="looking for a job",
                         memory_retriever=create_new_memory_retriever(),
                         llm=LLM,
                         memory=tommies_memory
                         )

# The current "Summary" of a character can't be made because the agent hasn't made
# any observations yet.
print(tommie.get_summary())

# import os
# from langchain.llms import OpenAI
# from langchain import PromptTemplate, LLMChain

# import gymnasium as gym
# import inspect
# import tenacity

# from langchain.schema import (
#     AIMessage,
#     HumanMessage,
#     SystemMessage,
#     BaseMessage,
# )
# from langchain.output_parsers import RegexParser


# class GymnasiumAgent():
#     @classmethod
#     def get_docs(cls, env):
#         return env.unwrapped.__doc__

#     def __init__(self, model, env):
#         self.model = model
#         self.env = env
#         self.docs = self.get_docs(env)

#         self.instructions = """
# Your goal is to maximize your return, i.e. the sum of the rewards you receive.
# I will give you an observation, reward, terminiation flag, truncation flag, and the return so far, formatted as:

# Observation: <observation>
# Reward: <reward>
# Termination: <termination>
# Truncation: <truncation>
# Return: <sum_of_rewards>

# You will respond with an action, formatted as:

# Action: <action>

# where you replace <action> with your actual action.
# Do nothing else but return the action.
# """
#         self.action_parser = RegexParser(
#             regex=r"Action: (.*)",
#             output_keys=['action'],
#             default_output_key='action')

#         self.message_history = []
#         self.ret = 0

#     def random_action(self):
#         action = self.env.action_space.sample()
#         return action

#     def reset(self):
#         self.message_history = [
#             SystemMessage(content=self.docs),
#             SystemMessage(content=self.instructions),
#         ]

#     def observe(self, obs, rew=0, term=False, trunc=False, info=None):
#         self.ret += rew

#         obs_message = f"""
# Observation: {obs}
# Reward: {rew}
# Termination: {term}
# Truncation: {trunc}
# Return: {self.ret}
#         """
#         self.message_history.append(HumanMessage(content=obs_message))
#         return obs_message

#     def _act(self):
#         act_message = self.model(self.message_history)
#         self.message_history.append(act_message)
#         action = int(self.action_parser.parse(act_message.content)['action'])
#         return action

#     def act(self):
#         try:
#             for attempt in tenacity.Retrying(
#                 stop=tenacity.stop_after_attempt(2),
#                 wait=tenacity.wait_none(),  # No waiting time between retries
#                 retry=tenacity.retry_if_exception_type(ValueError),
#                 before_sleep=lambda retry_state: print(
#                     f"ValueError occurred: {retry_state.outcome.exception()}, retrying..."),
#             ):
#                 with attempt:
#                     action = self._act()
#         except tenacity.RetryError as e:
#             action = self.random_action()
#         return action


# env = gym.make("Blackjack-v1")
# llm = OpenAI(openai_api_base="http://127.0.0.1:5001/v1",
#              openai_api_key="cvxcvcxvcxv")
# agent = GymnasiumAgent(model=llm, env=env)


# observation, info = env.reset()
# agent.reset()

# obs_message = agent.observe(observation)
# print(obs_message)

# while True:
#     action = agent.act()
#     observation, reward, termination, truncation, info = env.step(action)
#     obs_message = agent.observe(
#         observation, reward, termination, truncation, info)
#     print(f'Action: {action}')
#     print(obs_message)

#     if termination or truncation:
#         print('break', termination, truncation)
#         break
# env.close()
