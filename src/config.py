import os

# Set the project dir path.
project_dir_path = ''

# Set the API tokens.
hf_access_token = ''

# Set the Google and OpenAI API keys.
google_api_key = ''
openai_api_key = ''

# Set the HuggingFace model path for fine-tuning.
base_model_name = 'mistralai/Ministral-3-8B-Instruct-2512-BF16'

# Open and set the custom Greek and default chat templates.
template_path = os.path.join(project_dir_path, 'greek_chat_template.jinja')
with open(template_path, 'r', encoding = 'utf-8') as f:
    custom_chat_template = f.read()

default_template_path = os.path.join(project_dir_path, 'chat_template.jinja')
with open(default_template_path, 'r', encoding = 'utf-8') as f:
    default_chat_template = f.read()

# Set the path for the local quantized models.
trained_path = os.path.join(project_dir_path, 'Maistros-8B-Instruct-4bit')
mistral_path = os.path.join(project_dir_path, 'Ministral-3-8B-Instruct-4bit')
eurollm_path = os.path.join(project_dir_path, 'EuroLLM-9B-Instruct-2512-4bit')

# Set the LLM names for evaluation.
model_names = [
    'gpt-5-mini-2025-08-07',
    'gemini-3-flash-preview',
    'Ministral-3-8B-Instruct-4bit',
    'Maistros-8B-Instruct-4bit',
    'ilsp/Llama-Krikri-8B-Instruct',
    'Qwen/Qwen3-8B',
    'EuroLLM-9B-Instruct-2512-4bit',
    'TheFinAI/plutus-8B-instruct',
    'google/gemma-3n-E4B-it'
]

# Set the dataset names for experiments.
datasets_mc = [
    'DemosQA',
    'greek_pcr',
    'include',
    'mcqa_greek_asep',
    'medical_mcqa_greek',
    'plutus_qa_test',
    'truthful_qa_greek',
    'greek_mmlu_greek_specific'
]
 
datasets_oe = [
    'CulturaQA'
]
