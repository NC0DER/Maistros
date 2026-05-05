import os
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Mistral3ForConditionalGeneration,
    BitsAndBytesConfig
)
from huggingface_hub import HfApi
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig
from src.utils import preprocess_hf_dataset_to_messages
from src.config import (
    project_dir_path, 
    base_model_name,
    custom_chat_template,
    hf_access_token
)


def fine_tune_conversational_QA():
    """
    This function performs LoRA fine-tuning of a Large Language Model (LLM) for conversational QA.

    LoRA configuration:
        - The lora alpha parameter (α) is set to 2*r(ank), to achieve a scaling factor of 2.
        - We set rank = 32, alpha = 64 due to the small number of training samples.
        - The dropout is set to 0.1 to avoid overfitting.
        - The task type is set to "Causal_LM" because we fine-tune for Generative QA.
        - We use fan-out scaling to better match the frozen models' weight scales.
        - The bias is set to none due to fine-tuning for a simple QA task with a small rank.
        - We target most of the initial transformer layers for adaptation.
        - We save only the adapter modules.

    Supervised Fine-Tuning Configuration:
        - Random seed: 42
        - The max training sequence length is set to 3269.
        - We train for 4 epochs and save a model checkpoint after each epoch.
        - The train batch size is set to 2 to reduce VRAM with gradient accumulation steps to 8.
        - This means that the effective batch size is 16.
        - To reduce VRAM, we also use gradient checkpointing.
        - We use the adamw_8bit optimizer for LoRA fine-tuning.
        - We set the learning rate to 2e-5.
        - We utilize a mixed precision (16-bit) training environment.
        - The weight decay is set to 0.01 for stable LoRA fine-tuning.
        - We utilize the default ADAM hyperparameters.
        - The max gradient normalization clipping value is set to 1.0.
        - The warmup steps are 62.
        - Total steps are ⌊2000 training samples / 16 (effective batch size) * 4 epochs⌋
        - We decrease learning rate with cosine.
        - We do not shuffle or remove unused columns from the dataset, this was done earlier.
        - We fine-tune for conversational (assistant) QA not prompt completion.
        - We keep the checkpoints locally.
        - We format the train rows based on the model chat template.
        - Tokenization and padding are applied internally from the SFTTrainer object.
    """

    # Set the input paths for the training and validation datasets.
    train_dataset_path = os.path.join(project_dir_path, 'train.csv')
    val_dataset_path = os.path.join(project_dir_path, 'val.csv')

    # Load the training and validation datasets.
    train_dataset = load_dataset(
        'csv', 
        data_files = train_dataset_path,
        split = 'train'
    )

    val_dataset = load_dataset(
        'csv', 
        data_files = val_dataset_path,
        split = 'train' # Default HF split is called "train", but this is the validation split.
    )

    # Pre-process the train and validation datasets.
    train_dataset = train_dataset.map(
        preprocess_hf_dataset_to_messages,
        remove_columns = train_dataset.column_names
    )

    val_dataset = val_dataset.map(
        preprocess_hf_dataset_to_messages,
        remove_columns = val_dataset.column_names
    )

    # Set the directory path to save LoRA adapters.
    output_dir = os.path.join(project_dir_path, 'GreekQA-LoRA-adapters')

    # Set the base model for fine-tuning.
    model_name = base_model_name

    # Load the model tokenizer and set a custom chat template.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.chat_template = custom_chat_template

    # Causal LMs predict tokens from left to right and use EOS token for padding.
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # Utilize 4-bit normal float double quantization.
    # Mixed precision is set to bfloat16. 
    quantization_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = 'nf4',
        bnb_4bit_use_double_quant = True,
        bnb_4bit_compute_dtype = torch.bfloat16,
    )

    # Load the LM for LoRA fine_tuning using the above quantization scheme. 
    language_model = Mistral3ForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config = quantization_config,
        device_map = 'auto',
    )

    # Specify the LoRA configuration.
    lora_config = LoraConfig(
        r = 32,
        lora_alpha = 64,
        lora_dropout = 0.1,
        bias = 'none',
        task_type = 'CAUSAL_LM', # TaskType.CAUSAL_LM
        fan_in_fan_out = 'False',
        target_modules = [
            'q_proj',
            'k_proj',
            'v_proj',
            'o_proj',
            'gate_proj',
            'up_proj',
            'down_proj',
        ],
        modules_to_save = None
    )

    # Specify the parameters for the supervised fine-tuning.
    training_parameters = SFTConfig(
        seed = 42,
        dataset_kwargs = {
            'max_length': 3269,
            'truncation': True,
        },
        max_length = 3269,
        output_dir = output_dir,
        do_train = True,
        do_eval = True,
        per_device_train_batch_size = 2,
        per_device_eval_batch_size = 2,
        save_strategy = 'epoch',
        save_total_limit = 5,
        num_train_epochs = 4,
        logging_first_step = True,
        logging_steps = 25,
        eval_steps = 25,
        logging_strategy = 'steps',
        eval_strategy = 'steps',
        gradient_accumulation_steps = 8,
        gradient_checkpointing = True,
        optim = 'adamw_8bit',
        learning_rate = 2e-5,
        bf16 = True,
        fp16 = False,
        weight_decay = 0.1,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        adam_epsilon = 1e-8,
        max_grad_norm = 1.0,
        warmup_steps = 62,
        lr_scheduler_type = 'cosine',
        gradient_checkpointing_kwargs = {'use_reentrant': False},
        assistant_only_loss = True,
        completion_only_loss = False,
        packing = False,
        remove_unused_columns = False,
        shuffle_dataset = False,
        push_to_hub = False,   
        report_to = 'none'
    )

    # Define the trainer object and its parameters.
    # This object internally performs tokenization and max length padding.
    trainer = SFTTrainer(
        model = language_model,
        processing_class = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        peft_config = lora_config,
        args = training_parameters
    )

    trainer.train()

    return


def merge_adapter_weights_to_base_and_save():
    """
    This function loads the base model, its tokenizer and the adapted model weights.
    Then it merges the base model with the adapted model weights.
    Finally, it saves the new merged model and the original tokenizer.
    """
    # Set the input dir for the LoRA adapted weights.
    # and the output dir to save the merged model and its tokenizer.
    input_dir = os.path.join(project_dir_path, 'GreekQA-LoRA-adapters', 'checkpoint-375')
    output_dir = os.path.join(project_dir_path, 'Maistros-8B-Instruct')

    # Load the base model and its tokenizer.
    model_name = base_model_name
    base_model = Mistral3ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype = torch.bfloat16,
        device_map = 'auto',
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set a custom chat template for Greek QA SFT.
    tokenizer.chat_template = custom_chat_template

    # Load the adapters and merge them to the base model.
    lora_adapters = PeftModel.from_pretrained(base_model, input_dir)
    merged_model = lora_adapters.merge_and_unload()

    # Save the new merged model and its tokenizer.
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    return

def quantize_and_save_model(model_type: str):
    """
    This function loads and saves a BF16 model in 4-bit quantization.

    Parameters
    -----------
    model_type: the model type to save (str).

    Returns
    -----------
    None.
    """
    if model_type == 'merged':
        input_dir = os.path.join(project_dir_path, 'Maistros-8B-Instruct')
        output_dir = os.path.join(project_dir_path, 'Maistros-8B-Instruct-4bit')
    elif model_type == 'base':
        input_dir = base_model_name
        output_dir = os.path.join(project_dir_path, 'Ministral-3-8B-Instruct-4bit')
    elif model_type == 'EuroLLM': 
        # LLMs (>= 9B) do not fit in a 16GB VRAM GPU.
        # So we load them in larger card and save them quantized. 
        input_dir = 'utter-project/EuroLLM-9B-Instruct-2512'
        output_dir = os.path.join(project_dir_path, 'EuroLLM-9B-Instruct-2512-4bit')

    if model_type == 'EuroLLM':
        # Load the model tokenizer for EuroLLM.
        tokenizer = AutoTokenizer.from_pretrained(input_dir)
    else:
        # Load the model tokenizer for Ministral and Maistros.
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Set the custom chat template, if needed.
    if model_type == 'merged': 
        tokenizer.chat_template = custom_chat_template

    # Define the 4-bit quantization configuration
    quantization_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = 'nf4',
        bnb_4bit_use_double_quant = True,
        bnb_4bit_compute_dtype = torch.bfloat16
    )
    
    # Load the merged model from the input directory using bitsandbytes.
    if model_type == 'EuroLLM':
        quantized_model = AutoModelForCausalLM.from_pretrained(
            input_dir,
            quantization_config = quantization_config,
            device_map = 'auto',
        )
    else:
        quantized_model = Mistral3ForConditionalGeneration.from_pretrained(
            input_dir,
            quantization_config = quantization_config,
            device_map = 'auto',
        )

    # Save the quantized model.
    quantized_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
        
    return


def upload_model(model_name: str):
    """
    This function loads the full model into CPU, shards it into 5 GB chunks,
    so it can fit into memory limited GPUs with offloading, and uploads it to Huggingface.
    """
    model_directory = os.path.join(project_dir_path, model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_directory)

    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_directory, 
        device_map = 'cpu',
        low_cpu_mem_usage = True
    )

    tokenizer.push_to_hub(
        f'IMISLab/{model_name}',
        token = hf_access_token
    )

    model.push_to_hub(
        f'IMISLab/{model_name}', 
        max_shard_size = '5GB',
        token = hf_access_token
    )

    return


def upload_quantized_model(model_name: str):
    """
    Function to upload the quantized model directly to HuggingFace, 
    since push_to_hub() has not yet implemented code logic 
    that safely converts quantized Mistral 3 weights.
    """
    api = HfApi()
    model_directory = os.path.join(project_dir_path, model_name)
    
    api.upload_folder(
        folder_path = model_directory,
        repo_id = f'IMISLab/{model_name}',
        repo_type = 'model',
        token = hf_access_token
    )

    return
