import os
import re
import random

from datasets import load_dataset, concatenate_datasets
from src.config import project_dir_path, hf_access_token
from src.utils import extract_letter

# Define the output dir and create if it does not exist.
OUTPUT_DIR = os.path.join(project_dir_path, 'processed_datasets')
os.makedirs(OUTPUT_DIR, exist_ok = True)

# Define the strict column order you want for all datasets
TARGET_COLUMNS = ['id', 'question', 'answers', 'best_answer', 'best_answer_index']

def process_truthful_qa():
    """
    Loads the Truthful QA Greek dataset, processes it, and transforms it to the proper format.
    """
    truthful_qa = load_dataset('ilsp/truthful_qa_greek', data_dir = 'multiple_choice', split = 'all')
    
    truthful_qa = truthful_qa.remove_columns([
        'mc2_targets', 'question_en', 'mc1_targets_en', 
        'mc2_targets_en', 'question_mt', 'mc1_targets_mt', 'mc2_targets_mt'
    ])

    truthful_qa = truthful_qa.rename_column('mc1_targets', 'answers')
    
    ids = [i for i in range(len(truthful_qa))]
    truthful_qa = truthful_qa.add_column('id', ids)
    truthful_qa = truthful_qa.map(process_truthful_qa_rows, load_from_cache_file = False)

    truthful_qa = truthful_qa.select_columns(TARGET_COLUMNS)
    truthful_qa.to_csv(os.path.join(OUTPUT_DIR, 'truthful_qa_greek.csv'), index = False)


def process_truthful_qa_rows(row: dict) -> dict:
    """
    Transforms each row of the Truthful QA Greek dataset.
    """
    answers = row['answers']['choices']
    labels = row['answers']['labels']
    best_answer = answers[labels.index(1)]

    random.shuffle(answers)

    answers_column_text = ''
    letters = 'ΑΒΓΔΕΖΘΗΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ'

    for ans, j in zip(answers, letters):
        processed_answer = ' '.join(ans.split())
        answers_column_text += f'{j}. "{processed_answer}"\n\n'

        if ans == best_answer:
            best_answer_index = j

    answers_column_text = answers_column_text[:-2]

    row['answers'] = answers_column_text
    row['best_answer'] = best_answer
    row['best_answer_index'] = best_answer_index

    return row


def process_medical_mcqa_greek():
    """
    Loads the Medical MCQA dataset, processes it, and transforms it to the proper format.
    """
    medical_mcqa_greek = load_dataset('ilsp/medical_mcqa_greek', split = 'validation')
    
    medical_mcqa_greek = medical_mcqa_greek.remove_columns(['multiple_choice_scores', 'subject'])
    medical_mcqa_greek = medical_mcqa_greek.rename_column('inputs', 'question')
    medical_mcqa_greek = medical_mcqa_greek.rename_column('multiple_choice_targets', 'answers')
    medical_mcqa_greek = medical_mcqa_greek.rename_column('targets', 'best_answer')
    medical_mcqa_greek = medical_mcqa_greek.rename_column('idx', 'id')
    
    medical_mcqa_greek = medical_mcqa_greek.map(process_medical_mcqa_greek_rows, load_from_cache_file = False)
    
    medical_mcqa_greek = medical_mcqa_greek.select_columns(TARGET_COLUMNS)
    medical_mcqa_greek.to_csv(os.path.join(OUTPUT_DIR, 'medical_mcqa_greek.csv'), index = False)


def process_medical_mcqa_greek_rows(row: dict) -> dict:
    """
    Transforms each row of the Medical MCQA Greek dataset.
    """
    best_answer = row['best_answer']
    row['best_answer_index'] = extract_letter(best_answer[0])
    row['best_answer'] = row['best_answer'][0][3:]
    row['answers'] = '\n\n'.join([f'{answer[:3]}"{answer[3:]}"' for answer in row['answers']])

    return row


def process_mcqa_greek_asep():
    """
    Loads the MCQA Greek ASEP dataset, processes it, and transforms it to the proper format.
    """
    mcqa_greek_asep = load_dataset('ilsp/mcqa_greek_asep', split = 'default')
    
    mcqa_greek_asep = mcqa_greek_asep.remove_columns(['subject'])
    mcqa_greek_asep = mcqa_greek_asep.rename_column('choices', 'answers')
    mcqa_greek_asep = mcqa_greek_asep.rename_column('answer_text', 'best_answer')
    mcqa_greek_asep = mcqa_greek_asep.rename_column('answer', 'best_answer_index')
    
    mcqa_greek_asep = mcqa_greek_asep.map(process_mcqa_greek_asep_rows, load_from_cache_file = False)
    
    mcqa_greek_asep = mcqa_greek_asep.select_columns(TARGET_COLUMNS)
    mcqa_greek_asep.to_csv(os.path.join(OUTPUT_DIR, 'mcqa_greek_asep.csv'), index = False)


def process_mcqa_greek_asep_rows(row: dict) -> dict:
    """
    Transforms each row of the MCQA Greek ASEP dataset.
    """
    row['best_answer'] = row['best_answer'][3:]
    for i, answer in enumerate(row['answers']):
        row['answers'][i] = ['Α', 'Β', 'Γ', 'Δ'][i] + f'. "{answer[3:]}"'
    
    row['answers'] = '\n\n'.join(row['answers'])
    row['best_answer_index'] = ['Α', 'Β', 'Γ', 'Δ'][row['best_answer_index']]
    row['id'] = row['id'].replace('.', '-').replace('/', '-')

    return row


def process_include():
    """
    Loads the greek subset of the belebele dataset, processes it, and transforms it to the proper format.
    """
    include = load_dataset('CohereLabs/include-base-44', 'Greek', split = 'test')
    
    include = include.remove_columns(['language', 'country', 'domain', 'subject', 'regional_feature', 'level'])
    include = include.rename_column('answer', 'best_answer_index')
    
    ids = [i for i in range(len(include))]
    include = include.add_column('id', ids)
    include = include.map(process_include_rows, load_from_cache_file = False)
    
    include = include.select_columns(TARGET_COLUMNS)
    include.to_csv(os.path.join(OUTPUT_DIR, 'include.csv'), index = False)


def process_include_rows(row: dict) -> dict:
    """
    Transforms each row of the belebele dataset.
    """
    best_answer_column = f"option_{'abcd'[int(row['best_answer_index'])]}"
    
    row['best_answer'] = row[best_answer_column]
    row['best_answer_index'] = 'ΑΒΓΔ'[int(row['best_answer_index'])]
    row['answers'] = (
        f'Α. "{row['option_a']}"\n\n'
        f'Β. "{row['option_b']}"\n\n'
        f'Γ. "{row['option_c']}"\n\n'
        f'Δ. "{row['option_d']}"'
    )

    return row


def process_plutus_qa():
    """
    Loads the test split of the Plutus QA dataset, processes it, and transforms it to the proper format.
    """
    plutus_data = load_dataset('TheFinAI/plutus-QA', split = 'test')
    
    plutus_data = plutus_data.rename_column('query', 'question')
    
    # Add the ID column
    ids = [i for i in range(len(plutus_data))]
    plutus_data = plutus_data.add_column('id', ids)
    
    # Pre-allocate the missing columns with empty strings so the schema is forced to update.
    empty_strings = [""] * len(plutus_data)
    plutus_data = plutus_data.add_column('answers', empty_strings)
    plutus_data = plutus_data.add_column('best_answer', empty_strings)
    plutus_data = plutus_data.add_column('best_answer_index', empty_strings)
    
    # Run the map function.
    plutus_data = plutus_data.map(process_plutus_qa_rows, load_from_cache_file = False)
    
    # Select the target columns. 
    plutus_data = plutus_data.select_columns(TARGET_COLUMNS)
    plutus_data.to_csv(os.path.join(OUTPUT_DIR, 'plutus_qa_test.csv'), index = False)


def process_plutus_qa_rows(row: dict) -> dict:
    """
    Transforms each row of the Plutus QA dataset into the standard evaluation format.
    """
    raw_text = row['question']
    
    clean_question = raw_text.split('Ερώτηση:')[1].split('Πιθανές απαντήσεις:')[0].strip()
    row['question'] = clean_question
        
    raw_answers = raw_text.split('Πιθανές απαντήσεις:')[1].split('Απάντηση:')[0].strip()
    
    options_text = re.split(r'[ΑΒΓΔΕ]\)', raw_answers)[1:]
    options_text = [opt.strip() for opt in options_text]
    
    gold_index = int(row['gold'])
    letters = 'ΑΒΓΔΕ'
    row['best_answer_index'] = letters[gold_index]
    row['best_answer'] = options_text[gold_index]

    answers_column_text = ''
    for j, choice in zip(letters, options_text):
        answers_column_text += f'{j}. "{choice}"\n\n'
        
    row['answers'] = answers_column_text[:-2]

    return row


def process_demosqa():
    """
    Loads the DemosQA dataset, drops the unnecessary date column, 
    and saves it to the standard evaluation format.
    """
    demos_qa = load_dataset('IMISLab/DemosQA', split = 'test')
    
    demos_qa = demos_qa.select_columns(TARGET_COLUMNS)
    demos_qa.to_csv(os.path.join(OUTPUT_DIR, 'demos_qa.csv'), index = False)


def process_greek_pcr():
    """
    Loads the Greek PCR dataset, processes it, and transforms it to the proper format.
    """
    greek_pcr = load_dataset('ilsp/greek_pcr', split = 'default', token = hf_access_token)
    
    greek_pcr = greek_pcr.rename_column('prompt', 'question')
    
    greek_pcr = greek_pcr.map(process_greek_pcr_rows, load_from_cache_file = False)
    
    greek_pcr = greek_pcr.select_columns(TARGET_COLUMNS)
    greek_pcr.to_csv(os.path.join(OUTPUT_DIR, 'greek_pcr.csv'), index = False)


def process_greek_pcr_rows(row: dict) -> dict:
    """
    Transforms each row of the Greek PCR dataset into the standard evaluation format.
    """
    choices = [row['solution0'], row['solution1']]
    label = int(row['label'])
    
    letters = 'ΑΒ'
    
    row['best_answer'] = choices[label]
    row['best_answer_index'] = letters[label]
    
    answers_column_text = ''
    for j, choice in zip(letters, choices):
        processed_choice = ' '.join(choice.split()) 
        answers_column_text += f'{j}. "{processed_choice}"\n\n'
        
    row['answers'] = answers_column_text[:-2]
    
    return row


def process_greek_mmlu():
    """
    Loads specific subsets of the GreekMMLU dataset, processes them individually, 
    combines them into a single unified dataset, and saves it to a single CSV file.
    """
    subsets = [
        'Greek_History_Primary_School', 
        'Greek_History_Professional', 
        'Greek_History_Secondary_School', 
        'Greek_Literature', 
        'Greek_Mythology', 
        'Greek_Traditions',
        'Modern_Greek_Language_Primary_School',
        'Modern_Greek_Language_Secondary_School'
    ]
    
    # List to hold the processed datasets before merging
    processed_datasets = []
    
    for subset in subsets:
        # Load the test split for the current subset
        ds = load_dataset('dascim/GreekMMLU', subset, split = 'test')
        
        # Pre-allocate empty columns to force the schema to update correctly
        empty_strings = [''] * len(ds)
        ds = ds.add_column('answers', empty_strings)
        ds = ds.add_column('best_answer', empty_strings)
        ds = ds.add_column('best_answer_index', empty_strings)
        
        # Map the transformation (bypassing the cache).
        ds = ds.map(process_greek_mmlu_rows, load_from_cache_file = False)
        
        processed_datasets.append(ds)
        
    # Combine all individual datasets into one.
    unified_ds = concatenate_datasets(processed_datasets)
    
    # Add an ID column across the entire unified dataset.
    ids = [i for i in range(len(unified_ds))]
    unified_ds = unified_ds.add_column('id', ids)
    
    # Filter down to strictly our TARGET_COLUMNS.
    unified_ds = unified_ds.select_columns(TARGET_COLUMNS)
    
    # Export the dataset to a single .csv.
    file_path = os.path.join(OUTPUT_DIR, 'greek_mmlu_greek_specific.csv')
    unified_ds.to_csv(file_path, index = False)

    return


def process_greek_mmlu_rows(row: dict) -> dict:
    """
    Transforms each row of the GreekMMLU dataset into the standard evaluation format.
    """
    choices = row['choices']
    answer_idx = int(row['answer'])
    
    # The prompt specifies 2-4 options, so 4 letters are sufficient.
    letters = 'ΑΒΓΔ'
    
    # Extract the exact text and letter index of the correct answer.
    row['best_answer_index'] = letters[answer_idx]
    row['best_answer'] = choices[answer_idx]
    
    # Format the choices into a single string with Greek letters.
    answers_column_text = ''
    for j, choice in zip(letters, choices):
        # Clean up any potential extra whitespaces in the text.
        processed_choice = ' '.join(choice.split()) 
        answers_column_text += f'{j}. "{processed_choice}"\n\n'
        
    # Remove the trailing '\n\n' and assign to the answers column.
    row['answers'] = answers_column_text[:-2]
    
    return row


def process_all():

    process_plutus_qa()
    process_truthful_qa()
    process_include()
    process_mcqa_greek_asep()
    process_medical_mcqa_greek()
    process_demosqa()
    process_greek_pcr()
    process_greek_mmlu()
    
    return