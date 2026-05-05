import os
import pandas
import pathlib
import evaluate

from tqdm import tqdm
from src.llms import *
from src.prompts import *
from src.config import *
from src.utils import *


def generate_mc_answers(
    base_dir: str,
    dataset: pandas.DataFrame,
    dataset_name: str,
    model_name: str,
    model: GenAI
) -> pandas.DataFrame:
    """
    Generate multiple-choice answers using GenAI models.
    """

    # Sanitize model name to safely handle HuggingFace paths (e.g., "org/model" -> "org_model").
    safe_model_name = model_name.replace('/', '_')

    # Define directories.
    dataset_results_dir = os.path.join(base_dir, 'results', dataset_name)
    individual_save_dir = os.path.join(dataset_results_dir, safe_model_name)
    
    # Ensure individual model directory exists.
    pathlib.Path(individual_save_dir).mkdir(parents = True, exist_ok = True)

    # Set the instruction prompt.
    instruction_prompt = mcqa_instruction_prompt

    output_list = []

    for i in tqdm(range(len(dataset)), desc = 'Generating responses'):
        row = dataset.iloc[i]

        output_file_path = os.path.join(individual_save_dir, f"{row['id']}.csv")

        if os.path.exists(output_file_path):
            # Load cached result and append as a dict to keep data structures consistent.
            output_file = pandas.read_csv(output_file_path, index_col = False)
            row_dict = output_file.iloc[0].to_dict()
            output_list.append(row_dict)
            continue

        question = row['question']
        answers = row['answers']
        best_answer_index = row['best_answer_index']

        # Create the user prompt.
        user_prompt = (
            f'Ερώτηση: {question}\n\n'
            f'Διαθέσιμες Απαντήσεις:\n{answers}'
        )

        # Perform model inference.
        output = model.infer(
            user_prompt = user_prompt,
            instruction_prompt = instruction_prompt,
        )

        new_row = {
            'id': row['id'],
            'generated_answer': output,
            'correct_answer': best_answer_index,
        }

        # Save per-question result.
        pandas.DataFrame([new_row]).to_csv(
            output_file_path,
            encoding = 'utf-8',
            index = False,
        )

        output_list.append(new_row)

    # Aggregate CSV.
    generated_answers = pandas.DataFrame(output_list)

    if not generated_answers.empty:

        # Save aggregated file inside the dataset folder, alongside the individual model folders.
        aggregated_file_path = os.path.join(dataset_results_dir, f'{safe_model_name}-{dataset_name}.csv')
        generated_answers.to_csv(
            aggregated_file_path,
            encoding = 'utf-8',
            index = False,
        )

    return generated_answers


def run_mcqa_experiments():
    """
    Run multiple-choice QA experiments using GenAI models.
    """
    # Use the existing model generations to instantly reproduce the results.
    use_existing_generations = True

    # Ensure base results directory exists.
    results_dir = os.path.join(project_dir_path, 'results')
    pathlib.Path(results_dir).mkdir(parents = True, exist_ok = True)

    for model_name in model_names:

        # Initialize model to re-run generations, otherwise it is not required.
        if not use_existing_generations:
            model = GenAI(model_name)
        else:
            model = None
        
        safe_model_name = model_name.replace('/', '_')

        for dataset_name in datasets_mc:

            print(f'\n{dataset_name}\n')

            dataset_csv = os.path.join(project_dir_path, 'datasets', f'{dataset_name}.csv')
            if not os.path.exists(dataset_csv):
                raise FileNotFoundError(f'Dataset not found: {dataset_csv}')

            # Load original dataset.
            original_dataset = pandas.read_csv(dataset_csv, index_col = False)

            # Define the dataset's result directory.
            dataset_results_dir = os.path.join(results_dir, dataset_name)
            pathlib.Path(dataset_results_dir).mkdir(parents = True, exist_ok = True)

            # Define the aggregated file path.
            aggregated_file_path = os.path.join(dataset_results_dir, f'{safe_model_name}-{dataset_name}.csv')

            # Check if aggregated file already exists to skip generation.
            if not os.path.exists(aggregated_file_path):
                # Generate model answers.
                generate_mc_answers(
                    base_dir = project_dir_path,
                    dataset = original_dataset,
                    dataset_name = dataset_name,
                    model_name = model_name,
                    model = model
               )
            
            # Retrieve the generated dataset.
            generated_dataset = pandas.read_csv(aggregated_file_path, index_col = False)

            # Evaluate the model.
            accuracy_score = calculate_accuracy_score(
                original_dataset,
                generated_dataset,
            )

            print(f'{safe_model_name}: ', round(accuracy_score * 100, 2))

        # Delete model from VRAM to make room for the next one.
        del model
        torch.cuda.empty_cache()

    return


def generate_answers(
    base_dir: str,
    dataset: pandas.DataFrame,
    dataset_name: str,
    model_name: str,
    model: GenAI
) -> pandas.DataFrame:
    """
    Generate open-ended answers using GenAI models.
    """
    safe_model_name = model_name.replace('/', '_')
    dataset_results_dir = os.path.join(base_dir, 'results', dataset_name)
    individual_save_dir = os.path.join(dataset_results_dir, safe_model_name)
    
    pathlib.Path(individual_save_dir).mkdir(parents = True, exist_ok = True)

    # Set the instruction prompt for open-ended generation.
    instruction_prompt = oeqa_instruction_prompt

    output_list = []

    for i in tqdm(range(len(dataset)), desc = f'Generating responses ({safe_model_name})'):
        row = dataset.iloc[i]
        output_file_path = os.path.join(individual_save_dir, f"{row['id']}.csv")

        if os.path.exists(output_file_path):
            output_file = pandas.read_csv(output_file_path, index_col = False)
            row_dict = output_file.iloc[0].to_dict()
            output_list.append(row_dict)
            continue

        question = row['question']
        reference_answer = row['answer']

        user_prompt = f'Ερώτηση: {question}'

        output = model.infer(
            user_prompt = user_prompt,
            instruction_prompt = instruction_prompt,
        )

        new_row = {
            'id': row['id'],
            'question': question,
            'generated_answer': output,
            'reference_answer': reference_answer,
        }

        pandas.DataFrame([new_row]).to_csv(output_file_path, encoding = 'utf-8', index = False)
        output_list.append(new_row)

    generated_answers = pandas.DataFrame(output_list)

    if not generated_answers.empty:
        aggregated_file_path = os.path.join(dataset_results_dir, f'{safe_model_name}-{dataset_name}.csv')
        generated_answers.to_csv(aggregated_file_path, encoding = 'utf-8', index = False)

    return generated_answers


def evaluate_with_bertscore(
    base_dir: str,
    generated_dataset: pandas.DataFrame,
    dataset_name: str,
    model_name: str,
    bertscore_metric
) -> pandas.DataFrame:
    """
    Evaluate generated answers against gold answers using Hugging Face's BERTScore.
    """
    safe_model_name = model_name.replace('/', '_')
    dataset_results_dir = os.path.join(base_dir, 'evaluations', dataset_name)
    individual_eval_dir = os.path.join(dataset_results_dir, safe_model_name)
    
    pathlib.Path(individual_eval_dir).mkdir(parents = True, exist_ok = True)

    evaluated_list = []

    for i in tqdm(range(len(generated_dataset)), desc = f'Evaluating with BERTScore ({safe_model_name})'):
        row = generated_dataset.iloc[i]
        eval_file_path = os.path.join(individual_eval_dir, f"{row['id']}_eval.csv")

        # If already evaluated, load and skip
        if os.path.exists(eval_file_path):
            eval_file = pandas.read_csv(eval_file_path, index_col = False)
            evaluated_list.append(eval_file.iloc[0].to_dict())
            continue

        reference_answer = str(row['reference_answer'])
        generated_answer = str(row['generated_answer'])

        # Compute BERTScore using CPU for Greek text.
        results = bertscore_metric.compute(
            predictions = [generated_answer], 
            references = [reference_answer], 
            lang = 'el',
            device = 'cpu'
        )

        row_dict = row.to_dict()
        row_dict['bertscore_f1'] = results['f1'][0]
        row_dict['bertscore_precision'] = results['precision'][0]
        row_dict['bertscore_recall'] = results['recall'][0]

        # Save the individual result.
        pandas.DataFrame([row_dict]).to_csv(eval_file_path, encoding = 'utf-8', index = False)
        evaluated_list.append(row_dict)

    evaluated_dataset = pandas.DataFrame(evaluated_list)

    if not evaluated_dataset.empty:
        aggregated_eval_path = os.path.join(dataset_results_dir, f'{safe_model_name}-{dataset_name}_evaluated.csv')
        evaluated_dataset.to_csv(aggregated_eval_path, encoding = 'utf-8', index = False)

    return evaluated_dataset


def run_bertscore_experiments():
    """
    Run generation and BERTScore evaluation experiments.
    """
    # Use the existing model generations to instantly reproduce the results.
    use_existing_generations = True

    # Set the results and evaluations directories.
    results_dir = os.path.join(project_dir_path, 'results')
    evals_dir = os.path.join(project_dir_path, 'evaluations')
    
    pathlib.Path(results_dir).mkdir(parents = True, exist_ok = True)
    pathlib.Path(evals_dir).mkdir(parents = True, exist_ok = True)

    # Generate for each model.
    for model_name in model_names:
        safe_model_name = model_name.replace('/', '_')
        
        # Initialize model to re-run generations, otherwise it is not required.
        if not use_existing_generations:
            model = GenAI(model_name)
        else:
            model = None

        # Generate answers for the open-ended QA datasets.
        for dataset_name in datasets_oe:
            dataset_csv = os.path.join(project_dir_path, 'datasets', f'{dataset_name}.csv')
            original_dataset = pandas.read_csv(dataset_csv, index_col = False)

            dataset_results_dir = os.path.join(results_dir, dataset_name)
            pathlib.Path(dataset_results_dir).mkdir(parents = True, exist_ok = True)
            aggregated_file_path = os.path.join(dataset_results_dir, f'{safe_model_name}-{dataset_name}.csv')

            if not os.path.exists(aggregated_file_path):
                generate_answers(
                    base_dir = project_dir_path,
                    dataset = original_dataset,
                    dataset_name = dataset_name,
                    model_name = model_name,
                    model = model
               )
        
        del model
        torch.cuda.empty_cache()

    # Load the BERTScore metric once outside the loop.
    print('\nLoading BERTScore metric...')
    bertscore_metric = evaluate.load('bertscore')

    # Evaluate each model response.
    for model_name in model_names:
        safe_model_name = model_name.replace('/', '_')
        
        for dataset_name in datasets_oe:
            print(f'\n{dataset_name}\n')
            
            generated_file_path = os.path.join(results_dir, dataset_name, f'{safe_model_name}-{dataset_name}.csv')
            if not os.path.exists(generated_file_path):
                print(f'Skipping evaluation, missing generation file: {generated_file_path}')
                continue
                
            generated_dataset = pandas.read_csv(generated_file_path, index_col = False)

            dataset_eval_dir = os.path.join(evals_dir, dataset_name)
            pathlib.Path(dataset_eval_dir).mkdir(parents = True, exist_ok = True)
            aggregated_eval_path = os.path.join(dataset_eval_dir, f'{safe_model_name}-{dataset_name}_evaluated.csv')

            if not os.path.exists(aggregated_eval_path):
                evaluated_dataset = evaluate_with_bertscore(
                    base_dir = project_dir_path,
                    generated_dataset = generated_dataset,
                    dataset_name = dataset_name,
                    model_name = model_name,
                    bertscore_metric = bertscore_metric
                )
            else:
                evaluated_dataset = pandas.read_csv(aggregated_eval_path, index_col = False)

            # Calculate and print the macro F1 bertscore.
            macro_f1 = evaluated_dataset['bertscore_f1'].mean()
            print(f'BERTScore F1 for {safe_model_name}: {macro_f1:.4f}')

    return


def run_statistical_significance_tests():
    """
    Run statistical significance tests for the fine-tuned model against the baselines.
    """

    # Ensure base results and evaluation directories exist.
    results_dir = os.path.join(project_dir_path, 'results')
    evals_dir = os.path.join(project_dir_path, 'evaluations')

    pathlib.Path(results_dir).mkdir(parents = True, exist_ok = True)
    pathlib.Path(evals_dir).mkdir(parents = True, exist_ok = True)

    # Model baselines to check statistical significance against the fine-tuned model.
    # The state-of-the-art proprietary models are excluded from this comparison, 
    # since they outperform all open-source models. 
    baselines_list = [
        model_name.replace('/', '_') for model_name in model_names
        if model_name not in [
            'Maistros-8B-Instruct-4bit', 
            'gpt-5-mini-2025-08-07', 
            'gemini-3-flash-preview'
        ]
    ]

    fine_tuned_model_name = 'Maistros-8B-Instruct-4bit'

    for base_model_name in baselines_list:

        print(f'\nBase Model: {base_model_name}\n')
        
        # Verify statistical significance for multiple-choice datasets.
        for dataset_name in datasets_mc:

            print(f'\n{dataset_name}\n')

            dataset_csv = os.path.join(project_dir_path, 'datasets', f'{dataset_name}.csv')
            if not os.path.exists(dataset_csv):
                raise FileNotFoundError(f'Dataset not found: {dataset_csv}')

            # Load original dataset.
            original_dataset = pandas.read_csv(dataset_csv, index_col = False)

            # Define the dataset's result directory.
            dataset_results_dir = os.path.join(results_dir, dataset_name)
            pathlib.Path(dataset_results_dir).mkdir(parents = True, exist_ok = True)

            # Define the aggregated file paths for base and fine-tuned models.
            aggregated_file_path_base = os.path.join(dataset_results_dir, f'{base_model_name}-{dataset_name}.csv')
            aggregated_file_path_fine_tuned = os.path.join(dataset_results_dir, f'{fine_tuned_model_name}-{dataset_name}.csv')
    
            # Retrieve the generated datasets.        
            generated_dataset_base = pandas.read_csv(aggregated_file_path_base, index_col = False)
            generated_dataset_fine_tuned = pandas.read_csv(aggregated_file_path_fine_tuned, index_col = False)

            # Calculate the p_value using the McNemar binomial test.
            p_value = calculate_mcnemar(
                original_dataset,
                generated_dataset_base,
                generated_dataset_fine_tuned
            )

            # Print the results per dataset.
            print(f'{dataset_name}: P-Value: {p_value:.3f}')

            # Calculate the confidence intervals.
            bootstrap_accuracy_diff(original_dataset, generated_dataset_base, generated_dataset_fine_tuned)

        # Verify statistical significance for open-ended QA datasets.
        for dataset_name in datasets_oe:
            
            print(f'\n{dataset_name}\n')

            dataset_csv = os.path.join(project_dir_path, 'datasets', f'{dataset_name}.csv')
            if not os.path.exists(dataset_csv):
                raise FileNotFoundError(f'Dataset not found: {dataset_csv}')

            # Define the dataset's evaluations directory.
            dataset_evals_dir = os.path.join(evals_dir, dataset_name)
            pathlib.Path(dataset_evals_dir).mkdir(parents = True, exist_ok = True)
            
            # Define paths for the cached evaluated files.
            evaluated_file_path_base = os.path.join(dataset_evals_dir, f'{base_model_name}-{dataset_name}_evaluated.csv')
            evaluated_file_path_fine_tuned = os.path.join(dataset_evals_dir, f'{fine_tuned_model_name}-{dataset_name}_evaluated.csv')
    
            # Retrieve the generated datasets.        
            generated_dataset_base = pandas.read_csv(evaluated_file_path_base, index_col = False)
            generated_dataset_fine_tuned = pandas.read_csv(evaluated_file_path_fine_tuned, index_col = False)

            # Load the previously calculated scores for the base and fine-tuned models.
            evaluated_base_df = pandas.read_csv(evaluated_file_path_base, index_col = False)
            evaluated_fine_tuned_df = pandas.read_csv(evaluated_file_path_fine_tuned, index_col = False)
            
            # Convert the number to a numpy array.
            f1_base_scores = evaluated_base_df['bertscore_f1'].to_numpy()
            f1_fine_tuned_scores = evaluated_fine_tuned_df['bertscore_f1'].to_numpy()

            # Calculate the p_value using the Wilcoxon signed-rank test.
            p_value = calculate_wilcoxon_signed_rank(
                f1_base_scores,
                f1_fine_tuned_scores
            )

            # Print the results per dataset.
            print(f'{dataset_name}: P-Value: {p_value:.3f}')

            # Calculate the confidence intervals.
            calculate_bootstrap_bertscore(f1_base_scores, f1_fine_tuned_scores)

    return
