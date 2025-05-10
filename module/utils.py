import json
import sys
import os
from transformers import AutoTokenizer, AutoModelForMaskedLM
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
print(PROJECT_ROOT)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
hf_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")


import os
from dotenv import load_dotenv


load_dotenv()


api_key = os.getenv('API_KEY')


def load_questions(file_path):
    
    with open(file_path, 'r') as f:
        return json.load(f)

def send_openai_prompt(prompt_text, model_name="gpt-4o", temperature=0.7, top_p=0.7, token_limit=1000):
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt_text,
                }
        ],
            model="gpt-4o",
            temperature=temperature,
            #top_p=top_p,
            max_tokens=token_limit
        )
        #print(chat_completion.choices[0].message.content)
        return chat_completion.choices[0].message.content 
    except Exception as e:
        return f"Request failed: {e}"


def adaptive_openai_prompt(prompt_text, model_name="gpt-4o",):
    pass