import torch
import time
import random
from typing import Self
from openai import OpenAI
from google.genai import (
    types, 
    errors, 
    Client
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig, 
    Mistral3ForConditionalGeneration, 
    set_seed
)
from src.config import *

class GenAI:
    def __init__(
            self: Self, model_name: str, model_seed: int = 42, 
            temperature: float = 0.0, max_output_tokens: int = 1024) -> None:
        """
        Class constructor which initializes the OpenAI client to access GPT GenAI models.

        Parameters
        -----------
        self: the class Object (Self).
        model_name: the model path from HuggingFace (str).
        model_seed: the model seed (int).
        temperature: the model temperature, 0.0 corresponds to the most deterministic answers (float).

        Returns
        --------
        None.
        """
        # Initialize the client object using the model name and API key.
        self.model = None
        self.device = 'cuda'
        self.model_name = model_name
        self.model_seed = model_seed
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_daily_tokens = 245000 # Each day we receive these free daily tokens.

        match self.model_name:
            case name if name.startswith('gpt'):
                self.api_key = openai_api_key
                self.client = OpenAI(api_key = openai_api_key)
            
                # Initialize the model inference parameters
                self.parameters = {
                    'temperature': self.temperature,
                    'stream': False
                }

                # Additional parameters for GPT-5 reasoning models.
                # These reasoning models do not support the temperature setting.
                if self.model_name.startswith('gpt-5'):
                    reasoning = 'high'
                    if model_name.startswith('gpt-5-mini'):
                        reasoning = 'minimal'

                    self.parameters.update({
                        'reasoning': {'effort': reasoning},
                        'text': {'verbosity': 'low'},
                    })
                    del self.parameters['temperature']

            case name if name.startswith('gemini'):
                self.api_key = google_api_key
                self.client = Client(api_key = self.api_key)

                # Initialize the model inference parameters.
                self.parameters = {
                    'temperature': self.temperature,
                    'seed': self.model_seed,
                    'max_output_tokens': self.max_output_tokens
                }

            case name if name.startswith('Ministral') or name.startswith('Maistros') or name.startswith('EuroLLM'):

                if name.startswith('Maistros'):
                    path = trained_path
                elif name.startswith('Ministral'):
                    path = mistral_path
                elif name.startswith('EuroLLM'):
                    path = eurollm_path

                # Loading the model tokenizer.
                self.tokenizer = AutoTokenizer.from_pretrained(
                    path,
                    trust_remote_code = True
                )

                # Causal LMs predict tokens from left to right and use EOS token for padding.
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.padding_side = 'right'
                
                # Setting seed for reproducibility.
                set_seed(model_seed)

                # Loading the model with the necessary args.
                if name.startswith('EuroLLM'):
                    self.model = AutoModelForCausalLM.from_pretrained(
                        path,
                        device_map = self.device,
                        trust_remote_code = True
                    )
                else:    
                    self.model = Mistral3ForConditionalGeneration.from_pretrained(
                        path,
                        device_map = self.device,
                        trust_remote_code = True
                    )
                
                # Setting the model in evaluation mode.
                self.model.eval()

            case 'ilsp/Llama-Krikri-8B-Instruct' | 'Qwen/Qwen3-8B' | 'TheFinAI/plutus-8B-instruct' | 'google/gemma-3n-E4B-it':

                # Utilize 4-bit normal float double quantization.
                # Mixed precision is set to bfloat16. 
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit = True,
                    bnb_4bit_quant_type = 'nf4',
                    bnb_4bit_use_double_quant = True,
                    bnb_4bit_compute_dtype = torch.bfloat16,
                )

                # Loading the model tokenizer.
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code = True,
                    token = hf_access_token
                )

                # Causal LMs predict tokens from left to right and use EOS token for padding.
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.padding_side = 'right'

                # Setting seed for reproducibility.
                set_seed(model_seed)

                # Loading model with necessary args.
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config = quantization_config,
                    device_map = self.device,
                    trust_remote_code = True,
                    token = hf_access_token
                )
                
                # Setting the model in evaluation mode.
                self.model.eval()


    def infer(self: Self, user_prompt: str, instruction_prompt: str) -> str:
        """
        Class method that provides model inference for OpenAI models.

        Parameters
        -----------
        self: the class Object (Self).
        user_prompt: the user input prompt (str).
        instruction_prompt: the instruction prompt (str).

        Returns
        --------
        <object>: the decoded output (str).
        """
        match self.model_name:
            case name if name.startswith('gpt'):
                if self.max_daily_tokens > 0:
                    response = self.client.responses.create(
                        model = self.model_name,
                        input = user_prompt,
                        instructions = instruction_prompt,
                        **self.parameters
                    )
                    # Count the total tokens used during inference and subtract them from the max daily.
                    total_tokens = response.usage.total_tokens
                    self.max_daily_tokens -= total_tokens
                    print(f'Daily tokens left: {self.max_daily_tokens}')
                    decoded_output = response.output_text
                else:
                    raise ValueError('Free daily tokens have been consumed!')

            case name if name.startswith('gemini'):

                # Retry settings.
                max_retries = 5
                base_delay = 1 # Start waiting.

                for attempt in range(max_retries):
                    try:
                        config_kwargs = {
                            'system_instruction': instruction_prompt,
                            **self.parameters
                        }
                        if 'flash' in self.model_name:
                            config_kwargs['thinking_config'] = types.ThinkingConfig(thinking_level = 'minimal')
                        else:
                            config_kwargs['thinking_config'] = types.ThinkingConfig(thinking_level = 'low')

                        response = self.client.models.generate_content(
                            model = self.model_name,
                            contents = user_prompt,
                            config = types.GenerateContentConfig(**config_kwargs)
                        )
                        decoded_output = response.text
                        break # Break out of the retry loop on success.

                    except errors.ServerError as e:
                        # Check if it is a 503 or 249 (Too Many Requests).
                        if e.code in [503, 429]:
                            if attempt < max_retries - 1:
                                # Calculate wait time: 2, 4, 8, 16 seconds... (Exponential)
                                sleep_time = base_delay * (2 ** attempt)

                                sleep_time += random.uniform(0, 1)
                                print(f'Gemini overloaded ({e.code}). Retrying in {sleep_time:.2f}s...')
                                time.sleep(sleep_time)

                                continue
                            else:
                                # If we ran out of retries, raise the error.
                                raise e
                        else:
                            # If it is a different error (e.g., 400 Bad Request), crash immediately.
                            raise e
                    
            case name if name.startswith('Ministral') or name.startswith('Maistros'):
                
                # Defining the message format.
                messages = [
                    {'role': 'user', 'content': [{'type': 'text', 'text': '\n\n'.join((instruction_prompt, user_prompt))}]}
                ]

                # Applying the tokenizer chat template.
                tokenized = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt = True,  
                    return_tensors = 'pt', 
                    return_dict = True
                )

                # Sending the tokenized instances to the device.  
                tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
                input_len = len(tokenized['input_ids'][0])

                # Generating the model output.
                output = self.model.generate(
                    **tokenized,
                    max_new_tokens = self.max_output_tokens,
                    do_sample = False, # Equivalent to temperature = 0.0
                    temperature = None,
                    top_p = None,
                    top_k = None
                )

                # Decoding the assistant part of the output.
                decoded_output = self.tokenizer.decode(output[0][input_len:], skip_special_tokens = True)
        
            case 'ilsp/Llama-Krikri-8B-Instruct':

                # Defining the message format.
                messages = [
                    {'role': 'system', 'content': 'Είσαι το Κρικρί, ένα εξαιρετικά ανεπτυγμένο μοντέλο Τεχνητής Νοημοσύνης για τα ελληνικα και εκπαιδεύτηκες από το ΙΕΛ του Ε.Κ. "Αθηνά".'},
                    {'role': 'user', 'content': '\n\n'.join((instruction_prompt, user_prompt))},
                ]

                # Applying the tokenizer chat template.
                prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    add_generation_prompt = True, 
                    tokenize = False
                )

                # Tokenizing the inputs and send them to the device.
                input_prompt = self.tokenizer(
                    prompt, 
                    return_tensors = 'pt'
                ).to(self.device)

                # This works regardless of the transformers version or model type.
                if hasattr(input_prompt, 'input_ids'):
                    input_len = input_prompt.input_ids.shape[-1]
                elif isinstance(input_prompt, torch.Tensor):
                    input_len = input_prompt.shape[-1]
                else:
                    # Fallback for standard dicts.
                    input_len = input_prompt['input_ids'].shape[-1]

                # Generating the model outputs.
                outputs = self.model.generate(
                    input_prompt['input_ids'], 
                    max_new_tokens = self.max_output_tokens, 
                    do_sample = False, # Equivalent to temperature = 0.0
                    temperature = None,
                    top_p = None,
                    top_k = None
                )

                # Decoding the assistant part of the output.
                decoded_output = self.tokenizer.batch_decode(outputs[0][input_len:], skip_special_tokens = True)[0]
        
            case 'Qwen/Qwen3-8B':
                # Preparing the message format.
                messages = [
                    {'role': 'user', 'content': '\n\n'.join((instruction_prompt, user_prompt))}
                ]

                # Applying the chat template.
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize = False,
                    add_generation_prompt = True,
                    enable_thinking = False
                )

                # Tokenizing the inputs.
                model_inputs = self.tokenizer(
                    [text], 
                    return_tensors = 'pt'
                ).to(self.model.device)

                # Get the input length.
                input_len = len(model_inputs.input_ids[0])

                # Generating the output.
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens = self.max_output_tokens,
                    do_sample = False, # Equivalent to temperature = 0.0
                    temperature = None,
                    top_p = None,
                    top_k = None
                )

                # Decoding the assistant part of the output.
                output_ids = generated_ids[0][input_len:].tolist() 
                decoded_output = self.tokenizer.decode(output_ids, skip_special_tokens = True).strip('\n')

            case 'EuroLLM-9B-Instruct-2512-4bit':
                # Preparing the message format.
                messages = [
                    {'role': 'system', 'content': 'You are EuroLLM --- an AI assistant specialized in European languages that provides safe, educational and helpful answers.'},
                    {'role': 'user', 'content': '\n\n'.join((instruction_prompt, user_prompt))}
                ]

                # Applying the chat template.
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize = False,
                    add_generation_prompt = True
                )

                # Tokenizing the inputs.
                model_inputs = self.tokenizer(
                    text, 
                    return_tensors = 'pt'
                ).to(self.model.device)

                # This works regardless of the transformers version or model type.
                if hasattr(model_inputs, 'input_ids'):
                    input_len = model_inputs.input_ids.shape[-1]
                elif isinstance(model_inputs, torch.Tensor):
                    input_len = model_inputs.shape[-1]
                else:
                    # Fallback for standard dicts.
                    input_len = model_inputs['input_ids'].shape[-1]

                # Generating the output.
                outputs = self.model.generate(
                    **model_inputs,
                    max_new_tokens = self.max_output_tokens,
                    do_sample = False, # Equivalent to temperature = 0.0
                    temperature = None,
                    top_p = None,
                    top_k = None
                )

                # Decoding the assistant part of the output.
                decoded_output = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens = True)
            
            case 'TheFinAI/plutus-8B-instruct':

               # Defining the message format.
                messages = [
                    {'role': 'system', 'content': 'Είσαι το Plutus-8B, ένα εξαιρετικά ανεπτυγμένο μοντέλο Τεχνητής Νοημοσύνης για τα ελληνικα.'},
                    {'role': 'user', 'content': '\n\n'.join((instruction_prompt, user_prompt))},
                ]

                # Applying the tokenizer chat template.
                prompt = self.tokenizer.apply_chat_template(
                    messages, 
                    add_generation_prompt = True, 
                    tokenize = False
                )

                # Tokenizing the inputs and send them to the device.
                input_prompt = self.tokenizer(
                    prompt, 
                    return_tensors = 'pt'
                ).to(self.device)

                # This works regardless of the transformers version or model type.
                if hasattr(input_prompt, 'input_ids'):
                    input_len = input_prompt.input_ids.shape[-1]
                elif isinstance(input_prompt, torch.Tensor):
                    input_len = input_prompt.shape[-1]
                else:
                    # Fallback for standard dicts
                    input_len = input_prompt['input_ids'].shape[-1]

                # Generating the model outputs.
                outputs = self.model.generate(
                    input_prompt['input_ids'], 
                    max_new_tokens = self.max_output_tokens, 
                    do_sample = False, # Equivalent to temperature = 0.0
                    temperature = None,
                    top_p = None,
                    top_k = None
                )

                # Decoding the assistant part of the output.
                decoded_output = self.tokenizer.batch_decode(outputs[0][input_len:], skip_special_tokens = True)[0]

            case 'google/gemma-3n-E4B-it':
                # Constructing the model prompt and message format. 
                prompt = '\n\n'.join((instruction_prompt, user_prompt))
                messages = [
                    {'role': 'system', 'content': [{'type': 'text', 'text': 'You are a helpful assistant.'}]},
                    {'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}
                ]

                # Applying the chat template and tokenizing the input.
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt = True,
                    tokenize = True,
                    return_dict = True,
                    return_tensors = 'pt',
                ).to(self.model.device)

                # Measuring the input length. 
                input_len = inputs['input_ids'].shape[-1]

                # Generating the model output.
                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs, 
                        max_new_tokens = self.max_output_tokens, 
                        do_sample = False, # Equivalent to temperature = 0.0
                        temperature = None,
                        top_p = None,
                        top_k = None
                    )
                    outputs = outputs[0][input_len:]

                # Decoding the assistant part of the output.
                decoded_output = self.tokenizer.decode(outputs, skip_special_tokens = True)
        
        return decoded_output

