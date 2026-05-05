import os
import json
import pandas
import pathlib

from tqdm import tqdm
from statistics import mean
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import AutoTokenizer
from src.config import (
    project_dir_path,
    base_model_name,
)
from src.utils import generate_hash
from src.llms import GenAI
from src.prompts import *


def create_synthetic_data():
    """
    Utility function which loads a list of human curated greek keywords and their categories.
    This function then creates synthetic data in a directory based on the keywords,
    This function ignores previously created data.
        
    Arguments
    ---------
    None.

    Returns
    -------
    None.
    """
    # Initialize the GenAI model to create synthetic data with.
    LLM = GenAI('gpt-5')

    # Create the synthetic data path to save the samples.
    synthetic_data_path = os.path.join(project_dir_path, 'synthetic_data')
    
    # Make the synthetic data dir path if it does not already exists.
    pathlib.Path(synthetic_data_path).mkdir(exist_ok = True)

    # Load the greek keywords and their categories from the .json.
    greek_categories_keywords = []
    keywords_path = os.path.join(project_dir_path, 'QA_keywords.json')

    with open(keywords_path, 'r', encoding = 'utf-8') as f:
        category_keywords = json.load(f)
    
    for category, keywords in category_keywords.items():
        for keyword in keywords:
            greek_categories_keywords.append(
                [category, keyword]
            )

    # Find all synthetic data files from earlier runs and extract their ids.
    previous_ids = {
        p.name.split('_')[1] # Each file is saved as "<category>_<hash id>".
        for p in pathlib.Path(synthetic_data_path).iterdir()
        if p.is_file()
    }

    for category, keyword in tqdm(greek_categories_keywords):

        print(f'Generating questions for {category}: {keyword}')

        # Form the complete user prompt and generate the questions.
        complete_user_prompt = ''.join((user_keyword_prompt, f'"{keyword}"'))
        generated_questions = LLM.infer(complete_user_prompt, question_generation_system_prompt)

        # Process the generated questions and ignore the string before the first •.
        generated_questions = [
            generated_question.strip() 
            for generated_question in 
            generated_questions.split('•')[1:]
        ]
 
        print(f'Generated Questions\n\n------')
        print(generated_questions)

        if not generated_questions:
            raise ValueError('The question list should have been full!')
        
        for generated_question in generated_questions:
            
            # Compute a unique hash id based on the user question.
            hash_id = generate_hash(generated_question)

            if hash_id in previous_ids:
                continue

            # Form the file name and path and write the file.
            file_name = '_'.join((category, hash_id))
            file_path = os.path.join(synthetic_data_path, file_name)

            # Generate the answer for each question.
            complete_user_prompt = '\n\n'.join((user_answer_prompt, generated_question))
            generated_answer = LLM.infer(complete_user_prompt, question_answering_system_prompt)

            with open(file_path, 'w', encoding = 'utf-8', errors = 'ignore') as fd:
                fd.write('\n---\n'.join((generated_question, generated_answer)))

    return


def process_data_and_form_csv():
    """
    Utility function which loads all the synthetic data,
    then processes and collects them in a list.
    This list is then used to create a pandas dataframe 
    to be saved as a .csv file.
        
    Arguments
    ---------
    None.

    Returns
    -------
    None.
    """

    # Read the samples from the synthetic data directory.
    synthetic_data_path = os.path.join(project_dir_path, 'synthetic_data')

    # Find all synthetic data files from earlier runs and extract their ids.
    file_paths = [
        p.absolute()
        for p in pathlib.Path(synthetic_data_path).iterdir()
        if p.is_file()
    ]

    rows = []
    for file_path in tqdm(file_paths, desc = 'Processing and creating the dataset...'):
        with open(file_path, 'r', encoding = 'utf-8', errors = 'strict') as fd:
            synthetic_text = fd.read()

        question, answer = synthetic_text.split('\n---\n')

        # Replace the english question mark with the greek one.
        question = question.replace('?', ';').strip()

        # Remove the extra spaces generated at the end of lines.
        answer = answer.replace('  ', ' ')
        answer = answer.replace(' \n', '\n')
        
        # Extract the category and hash id from the file name.
        file_name = pathlib.Path(file_path).name
        category, hash_id = file_name.split('_')
    
        rows.append([hash_id, category, question, answer])

    # Create a dataframe from the list of lists.
    df = pandas.DataFrame(rows, columns = ['id', 'category', 'question', 'answer'])

    # Check if there are any duplicated values.
    duplicated_values = df.loc[df['id'].duplicated(), 'id']
    print(f'List of duplicates: {duplicated_values.unique()}')

    # Save the data into a .csv file.
    df.to_csv(
        os.path.join(project_dir_path, 'all_data.csv'), 
        index = False, encoding = 'utf-8', errors = 'strict'
    )
    
    return


def postprocess():
    """
    Function which postprocesses the dataset 
    and produces the train/val/test splits 
    using stratified sampling on the category label.
    """
    
    # Read the entire dataset.
    all_data_path = os.path.join(project_dir_path, 'all_data.csv')
    df = pandas.read_csv(
        all_data_path, index_col = False, 
        encoding = 'utf-8', encoding_errors = 'strict'
    )

    # Separate the input data from the category label.
    y = df.pop('category')
    X = df

    # Create train/validation/test splits using stratified sampling.
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify = y, test_size = 200, shuffle = True, random_state = 42)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, stratify = y_train, test_size = 500, shuffle = True, random_state = 42)

    # Join the input data and the category label for each split.
    X_train['category'] = y_train
    X_val['category'] = y_val
    X_test['category'] = y_test

    # Save the processed dataframes into .csv datasets.
    X_train.to_csv(os.path.join(project_dir_path, 'train.csv'), encoding = 'utf-8', index = False, errors = 'strict')
    X_val.to_csv(os.path.join(project_dir_path, 'val.csv'), encoding = 'utf-8', index = False, errors = 'strict')
    X_test.to_csv(os.path.join(project_dir_path, 'test.csv'), encoding = 'utf-8', index = False, errors = 'strict')
    
    return


def tokenizer_count():
    """
    This function counts and prints the amount of tokens
    of the training dataset (min, mean, max).
    """
    # Set the input path for the training dataset.
    input_dir = os.path.join(project_dir_path, 'train.csv')

    # Load the model tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Load and preprocess the training split of the dataset.
    training_dataset = load_dataset(
        'csv', 
        data_files = input_dir,
        split = 'train'
    )

    twok_count = 0
    counts = []
    for row in tqdm(training_dataset, desc = 'Counting token amounts for the train dataset...'):
        sample = '\n\n'.join((row['question'], row['answer']))

        # Encode each sample through the tokenizer and measure its length.
        encoded = tokenizer(sample, return_tensors = None, add_special_tokens = True)
        length = len(encoded['input_ids'])
        
        if length >= 2000:
            twok_count += 1

        counts.append(length)

    print(f'Minimum token amount: ', min(counts))
    print(f'Mean token amount: ', mean(counts))
    print(f'Maximum token amount: ', max(counts))
    print(twok_count)

    return
