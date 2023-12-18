import os
import re
import ast
import shlex
import requests
import subprocess
import pkg_resources
from sys import argv

from . import tools as t
from .prompts import MAIN_PROMPT
from .utils import download_ffmp
from .tool_mapping import generate_tools_mapping
from .interpretor import evaluate, extract_function_calls

tools = generate_tools_mapping()

def run_local(prompt):
    download_ffmp()
    ffmp_path = pkg_resources.resource_filename('ffmperative', 'bin/ffmp')
    safe_prompt = shlex.quote(prompt)
    command = '{} -p "{}"'.format(ffmp_path, safe_prompt)

    try:
        result = subprocess.run(command, capture_output=True, text=True, shell=True)

        output = result.stdout
        return output
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        return None

def run_remote(prompt):
    stop=["Task:"]
    complete_prompt = MAIN_PROMPT.replace("<<prompt>>", prompt.replace("'", "\\'").replace('"', '\\"'))
    headers = {"Authorization": f"Bearer {os.environ.get('HF_ACCESS_TOKEN', '')}"}
    inputs = {
        "inputs": complete_prompt,
        "parameters": {"max_new_tokens": 2048, "return_full_text": True, "stop":stop},
    }

    response = requests.post("https://api-inference.huggingface.co/models/bigcode/starcoder", json=inputs, headers=headers)
    if response.status_code == 429:
        logger.info("Getting rate-limited, waiting a tiny bit before trying again.")
        time.sleep(1)
        return run_remote(prompt)
    elif response.status_code != 200:
        raise ValueError(f"Error {response.status_code}: {response.json()}")

    result = response.json()[0]["generated_text"]
    for stop_seq in stop:
        if result.endswith(stop_seq):
            res = result[: -len(stop_seq)]
            answer = res.split("Answer:")[-1].strip()
            return answer 
    return result

def ffmp(prompt, remote=False, tools=tools):
    if remote:
        parsed_output = run_remote(prompt)
    else:
        parsed_output = run_local(prompt)
    if parsed_output:
        try:
            extracted_output = extract_function_calls(parsed_output, tools)
            parsed_ast = ast.parse(extracted_output)
            result = evaluate(parsed_ast, tools)
            return result
        except SyntaxError as e:
            print(f"Syntax error in parsed output: {e}")
    else:
        return None
