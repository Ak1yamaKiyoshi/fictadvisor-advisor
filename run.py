import os
import requests
from typing import List, Tuple, Dict, Optional, Union

from datetime import datetime
from colorama import Fore, init, Style
import asyncio
import json
from dotenv import load_dotenv
import numpy as np
import re

REPOSITORIES = "./assets/repositories/"
REPOSITORY = REPOSITORIES + "fictadvisor"
MODEL_NAME = "mistral-medium"
CONVERSATIONS = "./conversations/"

if not os.path.exists(CONVERSATIONS):
    os.makedirs(CONVERSATIONS)

load_dotenv()
mistral_api_key = os.getenv("MISTRAL_API_KEY")


class FileScrapper:
    def __init__(self) -> None:
        pass

    def listdir_recursive(self, path: str) -> List[str]:
        """ Returns all paths in directory. """
        contents = []
        for dirpath, dirnames, filenames in os.walk(path):
            for dirname in dirnames:
                contents.append(os.path.join(dirpath, dirname))
            for filename in filenames:
                contents.append(os.path.join(dirpath, filename))
        return contents

    def scrap_content(self, path: str, ignore_files: Optional[List[str]] = None) -> str:
        """ Scraps all content of files in given directory """
        contents = list()
        for directory in self.listdir_recursive(path):
            if "node_modules" in directory:
                continue
            if "package.json" in directory:
                continue
            if "yarn.lock" in directory:
                continue
            if directory and directory.split("/")[-1] not in ignore_files:
                try:
                    with open(directory, "r+") as f:
                        contents.append(f.read())
                except:
                    continue
        return contents

    def get_gitignore(self, path: str):
        with open(path + ".gitgnore", "r") as file:
            content = file.read()

        filenames = list()
        for line in content:
            if line.startswith("#"):
                continue
            filenames.append(line)
        return filenames

    def scrap_repository(self, path: str):
        try:
            ignore_files = self.get_gitignore(path)
        except:
            ignore_files = list()
        contents = self.scrap_content(path, ignore_files)
        return contents

    def find_file(self, original_path, filename):
        for root, dirs, files in os.walk(original_path):
            if filename in files:
                return os.path.join(root, filename)
        return None

    def scrap_file(self, path):
        with open(path, "r") as file:
            content = file.read()
        return content


class OpenAIWrapper:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")

    def request(
        self,
        model: str = "text-davinci-002",
        messages: List[dict[str, str]] = [{"role": "user", "content": "Who is"}],
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        stream: bool = False,
        safe_prompt: bool = True,
        random_seed: int = 0,
        api_key: str = "",
    ) -> dict:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        data = {
            "model": model,
            **({"temperature": temperature}),
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
            # "safe_prompt": safe_prompt,
            # "random_seed": random_seed,
        }
        return requests.post(url, json=data, headers=headers).json()

    def request_embeddings(
        self, api_key: str = "", document: List[str] = ["A", "B"],
    ) -> List[List[float]]:
        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        data = {
            "model": "text-embedding-ada-002",
            "input": document,
        }
        response = requests.post(url, json=data, headers=headers).json()
        return response["data"]


class OpenAIChatWrapper(OpenAIWrapper):
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        api_key: str = "",
        use_temperature: bool = True,
    ):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.use_temperature = use_temperature
        self.chat_history = None
        self.system_prompt = None

    def setup(self, system_prompt: Optional[str] = None) -> "OpenAIChatWrapper":
        if not system_prompt:
            system_prompt = "You are OpenAI. You are going to answer my questions. In addition to them, you will have chat history."
        self.system_prompt = system_prompt
        self.chat_history = [{"role": "system", "content": system_prompt}]
        return self

    def set_system_prompt(self, new: str) -> None:
        self.chat_history[0] = {"role": "system", "content": new}
        print(self.chat_history)

    def __prepare_history(self, query: str) -> List[dict[str, str]]:
        new_message = {"role": "user", "content": query}
        self.chat_history.append(new_message)
        history = self.chat_history.copy()
        for message in history:
            if message["role"] == "system":
                continue
            message["content"] = f"{message['role'].upper()}: {message['content']}"
            message["role"] = "user"
        return history

    def invoke(self, query: str) -> str:
        history = self.__prepare_history(query)
        response = super().request(
            model=self.model,
            messages=history,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key,
        )
        return response["choices"][0]["message"]["content"]

    def embed(self, document: List[str]) -> List[float]:
        document_embeddings = self.request_embeddings(
            api_key=self.api_key, document=document
        )
        print(f"{Fore.RED} {document_embeddings[0]['embedding']} {Fore.RESET}")
        return document_embeddings[0]["embedding"]


class MistralWrapper:
    def __init__(self):
        pass

    def request(
        self,
        model: str = "mistral-tiny",
        messages: List[dict[str, str]] = [
            {"role": "user", "content": "Who is the most renowned French painter?"}
        ],
        top_p: str = "null",
        temperature: float = 0.7,
        max_tokens: int = 32,
        stream: bool = False,
        safe_prompt: bool = False,
        random_seed: float = 0,
        api_key: str = "",
    ):
        """
            - [model] Id of the model
            - [messages] Example: [{"role": "user", "content": "Who is the most renowned French painter?"}]
            - [top_p] Nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
                We generally recommend altering this or temperature but not both.
            - [temperature] What sampling temperature to use, between 0.0 and 1.0. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. We generally recommend altering this or top_p but not both.
            - [max_tokens] The maximum number of tokens to generate in the completion.
                The token count of your prompt plus max_tokens cannot exceed the model's context length
            - [stream] Whether to stream back partial progress. If set, tokens will be sent as data-only server-sent events as they become available, with the stream terminated by a data: [DONE] message. Otherwise, the server will hold the request open until the timeout or until completion, with the response containing the full result as JSON.
            - [safe_prompt] Whether to inject a safety prompt before all conversations.
            - [random_seed] The seed to use for random sampling. If set, different calls will generate deterministic results.
        """

        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        data = {
            "model": model,
            **({"temperature": temperature} if top_p == "null" else {"top_p": top_p}),
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
            "safe_prompt": safe_prompt,
            "random_seed": random_seed,
        }
        return requests.post(url, json=data, headers=headers)

    def request_embeddings(self, api_key: str = "", document: List[str] = ["A", "B"]):
        url = "https://api.mistral.ai/v1/embeddings"
        data = {"model": "mistral-embed", "input": document, "encoding_format": "float"}
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        response = requests.post(url=url, json=data, headers=headers)
        if response.status_code == 200:
            return [result["embedding"] for result in response.json()["data"]]
        else:
            print("Error:", response.status_code, response.text)
            raise RuntimeError(response.text)

    def cosine_similarity(self, a, b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)

    def embedding_search(
        self, document: List[str], query: str, api_key: str = ""
    ) -> Tuple[str, float]:
        # document_embeddings: List[List[float]]
        document_embeddings = self.embedding_search(api_key, document)
        query_embedding = self.embedding_search(api_key, [query])[0]
        max_similarity = -1.0
        most_similar_sentence = ""

        for i, sentence_embedding in enumerate(document_embeddings):
            similarity = self.cosine_similarity(query_embedding, sentence_embedding)
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_sentence = document[i]

        return most_similar_sentence, max_similarity

    def embed(self, document: List[str], api_key: str = "") -> List[float]:
        document_embeddings = self.embedding_search(api_key, document)
        return document_embeddings


class MistralChatWrapper(MistralWrapper):
    def __init__(
        self,
        model: str = "mistral-small",
        top_p: float = 0.8,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        api_key: str = "",
        temperature_insead_top_p: bool = True,
    ) -> None:

        super().__init__()

        self.model = model

        self.temperature: float = temperature
        self.max_tokens: int = max_tokens
        self.top_p: int = top_p
        self.use_temperature: bool = temperature_insead_top_p
        self.api_key = api_key

        self.chat_history: List[Dict[str, str]] = None
        self.system_prompt: str = None

    def setup(self, system_prompt: Optional[str] = None) -> "MistralChatWrapper":
        if not system_prompt:
            system_prompt = "You are MistralAI. You are going to answer my questions. In addition to them, you will have chat history."

        self.system_prompt = system_prompt
        self.chat_history = [{"role": "system", "content": system_prompt}]
        return self

    def set_system_prompt(self, new: str) -> None:
        self.chat_history[0] = {"role": "system", "content": new}
        print(self.chat_history)

    def __prepare_history(self, query: str) -> List[Dict[str, str]]:
        # todo: Check system prompts, try to add them in each writing
        new_message = {"role": "user", "content": query}
        self.chat_history += [new_message]
        history = self.chat_history.copy()
        for message in history:
            if message["role"] == "system":
                continue
            message["content"] = f"{message['role'].upper()}: {message['content']}"
            message["role"] = "user"
        return history

    def invoke(self, query: str):
        history = self.__prepare_history(query)
        return self.request(
            self.model,
            temperature=self.temperature,
            messages=history,
            top_p=str(self.top_p) if not self.use_temperature else "null",
            max_tokens=self.max_tokens,
            api_key=self.api_key,
        ).json()["choices"][0]["message"]["content"]

    def embed(self, document: List[str],) -> List[float]:
        document_embeddings = self.request_embeddings(self.api_key, document)
        return document_embeddings


class WrapperLLM:
    def __init__(self, llm):
        self.llm = llm

    def invoke(self, query: str):
        return self.llm.invoke(query)

    @property
    def get_temperature(self):
        return self.llm.temperature

    @get_temperature.setter
    def temperature(self, new_temperature):
        self.llm.temperature = new_temperature

    @property
    def history(self):
        return self.llm.chat_history

    def set_system_prompt(self, new_system_prompt: str) -> None:
        self.llm.set_system_prompt(new_system_prompt)

    def get_embeddings(self, document):
        return self.llm.embed(document)

    def cosine_similarity(self, a, b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)

    def embedding_search(
        self,
        document: List[str],
        document_embeddings: List[float],
        query: str,
        threshold: float = 0.8,
    ) -> List[Tuple[str, float]]:
        query_embedding = self.get_embeddings([query])
        print(len(query_embedding), len(document_embeddings))
        results = []

        for i, sentence_embedding in enumerate(document_embeddings):
            similarity = self.cosine_similarity(query_embedding, sentence_embedding)
            if similarity > threshold:
                results.append(document[i])

        return results


def split_by_indentation(text):
    lines = text.split("\n")
    result = []
    current_block = []

    for line in lines:
        if line.strip():
            indentation = len(line) - len(line.lstrip())
            if current_block and indentation > len(current_block[-1]) - len(
                current_block[-1].lstrip()
            ):
                result.append("\n".join(current_block))
                current_block = [line]
            else:
                current_block.append(line)

    if current_block:
        result.append("\n".join(current_block))

    return result


def split_text_into_chunks(text, chunk_size):
    text_length = len(text)
    num_chunks = (text_length + chunk_size - 1) // chunk_size
    chunks = [text[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks)]
    return chunks


class Chat:
    def __init__(self):
        mistralLLM = MistralChatWrapper(
            api_key=mistral_api_key, temperature_insead_top_p=True,
        )
        mistralLLM.setup()
        self.llm = WrapperLLM(mistralLLM)

        # openaiLLM = OpenAIChatWrapper(api_key=os.getenv('OPENAI_API_KEY'))
        # openaiLLM.setup()
        # self.llm = WrapperLLM(openaiLLM)

    def run(self):
        print(
            f"""{Fore.LIGHTCYAN_EX} = Console Chat with LLM.
 = '/bye' - exit conversation
 = '/rollback' - rollback previous pair of answers (user and llm)
 = '/save <filename>' - save conversation in file
 = '/system <prompt>' - set system prompt
 = '/add <repository path>' - list directories
    '--confirm' - confirm adding repository
 = 't=float' - set new temperature
 _ '$embd{{filename}}' - at any place in prompt, will use embeddings from added repositories{Fore.RESET}
"""
        )
        repositories = list()
        repositories_content = list()
        query: Optional[str] = None
        prev_query: Optional[str] = None
        embedings: List[float] = None
        while True:
            prev_query = query
            current_time = datetime.now().strftime("%H:%M")
            query = input(
                f"{Style.BRIGHT}{Fore.LIGHTYELLOW_EX} [{current_time}] Query: {Fore.BLUE}"
            )
            print(f"{Style.NORMAL} {Fore.RESET}", end="")

            if query == "/bye":
                break
            # * Match File
            if "$embd" in query:
                while "$embd" in query:
                    if not repositories:
                        print(" [ -- ] There is no repositories added ")
                    match = re.search(r"\$embd\{(.*?)\}", query)
                    if match:
                        content = match.group(1)
                    else:
                        print(" [ -- ] No match found.")
                        continue
                    file_path = FileScrapper().find_file(repositories[-1], content)
                    file_content = FileScrapper().scrap_file(file_path)
                    print(f" [ -- ] File content lenght: {len(file_content)}")
                    file_content = split_text_into_chunks(file_content, 256)
                    print(f" [ -- ] Requested embeddings for file: {content}")
                    file_embeddings = self.llm.get_embeddings(file_content)
                    search = self.llm.embedding_search(
                        file_content, file_embeddings, query
                    )
                    query = query.replace("$embd", str(search), 1)
                    print(f" [ -- ] Found similarities: {len(search)}")
            elif query == "/rollback":
                self.llm.chat_history = self.llm.chat_history[:-2]
                continue
            elif query.startswith("t="):
                new_temperature = query.split("t=")[1]
                self.llm.set_temperature(new_temperature)
            elif query.startswith("/save"):
                filename = query.split("/save")[1].strip()
                with open(CONVERSATIONS + filename, "w") as f:
                    for message in self.llm.history:
                        f.write(f" > {message['role'].upper()}: {message['content']}\n")
                continue
            elif query.startswith("/system"):
                new_system_prompt = query.split("/system")[1].strip()
                self.llm.set_system_prompt(new_system_prompt)
                continue
            elif query.startswith("/add"):
                path = query.split("/add")[1].strip().split(" ")[0]
                confirm = "--confirm" in query
                print(f" [ -- ] List directories: {os.listdir(path)}")
                if confirm:
                    repositories.append(path)
                print(f" [ -- ] Added repositories: {repositories}")
                continue

            current_time = datetime.now().strftime("%H:%M")
            print(
                f"{Style.BRIGHT}{Fore.GREEN}[{current_time}] Answer: {Style.NORMAL}{Fore.RESET}{self.llm.invoke(query)}"
            )


#! Fix OpenAI embeddings
#! Browse Sites with embd `$embd{link|query}`
#! Add query to embd `$embd{filename|query}`
#! Cache file contents and embeddings
#! Add posibility to search needed files by embeddings of their names and query, and add this to $embd{}
#! Mistral Local
#! Ollama Local

Chat().run()
