import re
import math
import hashlib
import difflib
import pandas
import numpy
import nltk

from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import accuracy_score


def word_tokenize(text):
    """
    Function that tokenizes texts into a list of words.
    """
    return nltk.RegexpTokenizer(r'[ ,;;.!?:-]+', gaps = True).tokenize(text)


def measure_token_lengths(df, label):
    """
    Function that measures token lengths for a dataframe.
    """
    return [len(word_tokenize(text)) for text in df[label]]


def generate_hash(text: str) -> str:
    """
    This function encodes a text using SHA-3 512 bits (64 bytes).
    The encoded sequence is a string of 128 hexadecimal characters.
    
    Arguments
    ---------
    text: The text to be encoded (str).

    Returns
    -------
    <object>: the encoded sequence (str).
    """
    return hashlib.sha3_512(text.encode('utf-8')).hexdigest()


def preprocess_hf_dataset_to_messages(row):
    """
    This function formats a HF dataset to a messages conversational template.

    Arguments
    ---------
    row: the huggingface dataset row containing the id, question, answer and category.

    Returns
    -------
    <object>: the formatted messages object.
    """
    return {
        'messages': [
            {
                'role': 'user',
                'content': row['question']
            },
            {
                'role': 'assistant',
                'content': row['answer']
            }
        ]
    }


def extract_letter(answer: str) -> str:
    """
    Extracts the answer letter (Α-Ω) from the provided string.

    Parameters
    -----------
    answer (str): The answer text from which the letter should be extracted.

    Returns:
    -----------
    str: The extracted letter (Α-Ω) if found, otherwise 'No match'.
    """

    # Define a regex pattern to match the letter in the expected formats.
    # Search for single letters in range Α-Ω excluding the greek articles 'Ο' and 'Η'. 
    if (match := re.search(r'\b[Α-ΖΘ-ΞΠ-Ω]\b', answer)):
        return match.group(0)

    return 'No match'


def extract_letter_from_generated_answer(answer: str, possible_answers: str) -> str:
    """
    Extracts the answer letter (Α-Ω) from the generated answer using Regex and Fuzzy Matching.

    Parameters
    -----------
    answer (str): The answer text from which the letter should be extracted.
    possible_answers (str): A string containing all possible answers separated by double newlines.

    Returns
    --------
    str: The extracted letter (Α-Ω) if found, otherwise 'No match'.
    """
    # Skip erroneous NaN values.
    if isinstance(answer, float) and math.isnan(answer):
        answer = ''
    
    # Normalize whitespaces.
    answer = ' '.join(answer.split())

    # Safely extract and format options.
    options = [opt.strip() for opt in possible_answers.strip().split('\n\n') if opt.strip()]
    options_without_letter = [re.sub(r'^[Α-Ω][.:)]\s*[\'"]?', '', opt).strip(' "\'') for opt in options]

    # Case 1: Very short answer length.
    if len(answer) < 4:
        if (match := re.search(r'\b([Α-Ω])\b', answer)):
            return match.group(1)

    # Case 2: Exact option match.
    elif answer in options:
        return answer[0]
        
    # Case 3: Bold letter detection (e.g., **A**, **B.**).
    elif (match := re.search(r'\*\*([Α-Ω])[.:)]*\*\*', answer)):
        return match.group(1)

    # Case 4: Keyword phrase extraction.
    elif (match := re.search(r'[αΑ]πάντηση\s*(?:είναι\s*(?:η\s*)?)?(?:το\s*γράμμα\s*)?(?:η\s*επιλογή\s*)?[:.]?\s*[*"\'(]*([Α-Ω])\b', answer)):
        return match.group(1)

    # Case 5: Fuzzy text matching.
    else:
        answer_sentences = re.split(r'[.!?]\s+', answer.lower())
        
        best_ratio = 0.0
        best_match_index = -1
        
        for i, option_text in enumerate(options_without_letter):
            opt_lower = option_text.lower()
            
            # Case 5a: Substring match.
            if opt_lower in answer.lower():
                return options[i][0]
                
            # Case 5b: Sentence-level fuzzy matching.
            for sentence in answer_sentences:
                ratio = difflib.SequenceMatcher(None, opt_lower, sentence).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match_index = i
        
        # Threshold for acceptable fuzzy match.
        if best_ratio > 0.75:
            return options[best_match_index][0]

    # Condition 6: Most frequent letter fallback.
    matches = re.findall(r'\b([Α-Ω])[.:\)\'\"(\*+)]', answer)
    if matches:
        return max(matches, key = matches.count)

    return 'No match'


def calculate_accuracy_score(original_dataset: pandas.DataFrame, generated_dataset: pandas.DataFrame) -> float:
    """
    Function that calculates the accuracy scores for a dataset.

    Parameters
    -----------
    original_dataset: the dataset containing the gold answers.
    generated_dataset: the dataset containing the model generated answers.

    Returns
    --------
    accuracy: the accuracy score (float).
    """

    y_pred = [
        extract_letter_from_generated_answer(answer, options)
            for answer, options in zip(
                generated_dataset['generated_answer'], 
                original_dataset['answers']
            )
    ]

    y_true = generated_dataset['correct_answer'].to_list()

    accuracy = accuracy_score(y_true, y_pred)
    return accuracy


def calculate_mcnemar(
        original_dataset: pandas.DataFrame, 
        generated_dataset_base: pandas.DataFrame,
        generated_dataset_fine_tuned: pandas.DataFrame) -> float:
    """
    Function that calculates the McNemar binomial test scores 
    between a base and a fine_tuned model for a dataset.

    Parameters
    -----------
    original_dataset: the dataset containing the gold answers.
    generated_dataset_base: the dataset containing the base model's generated answers.
    generated_dataset_fine_tuned: the dataset containing the fine_tuned model's generated answers.

    Returns
    --------
    p_value: the p_value score (float).
    """

    y_base = numpy.array([
        extract_letter_from_generated_answer(answer, options)
            for answer, options in zip(
                generated_dataset_base['generated_answer'], 
                original_dataset['answers']
            )
    ])
        
    y_finetuned = numpy.array([
        extract_letter_from_generated_answer(answer, options)
            for answer, options in zip(
                generated_dataset_fine_tuned['generated_answer'], 
                original_dataset['answers']
            )
    ])

    y_true = numpy.array(generated_dataset_base['correct_answer'].to_list())

    # Create the 2x2 contingency table
    # Row 0: Base correct, Row 1: Base wrong
    # Col 0: FT correct,   Col 1: FT wrong
    
    both_correct = numpy.sum((y_base == y_true) & (y_finetuned == y_true))
    base_correct_ft_wrong = numpy.sum((y_base == y_true) & (y_finetuned != y_true))
    ft_correct_base_wrong = numpy.sum((y_base != y_true) & (y_finetuned == y_true))
    both_wrong = numpy.sum((y_base != y_true) & (y_finetuned != y_true))

    table = [
        [both_correct, base_correct_ft_wrong],
        [ft_correct_base_wrong, both_wrong]
    ]

    # We use Binomial distribution (exact = True).
    p_value = mcnemar(table, exact = True).pvalue

    return p_value


def calculate_wilcoxon_signed_rank(
        f1_base_scores: numpy.typing.NDArray,
        f1_fine_tuned_scores: numpy.typing.NDArray) -> float:
    """
    Function runs a Wilcoxon signed-rank test between 
    the BERTScore F1 scores of a base and a fine_tuned model for a dataset.

    Parameters
    -----------
    f1_base_scores: the numpy array containing the f1 scores for the base model.
    f1_fine_tuned_scores: the numpy array containing the f1 scores for the generated model.

    Returns
    --------
    p_value: the p_value score (float).
    """

    # Run the Wilcoxon signed-rank test using the zero method wilcox 
    # to handle the case of these models achieving the same score.
    _, p_value = wilcoxon(f1_base_scores, f1_fine_tuned_scores, zero_method = 'wilcox')

    return p_value


def bootstrap_accuracy_diff(
        original_dataset: pandas.DataFrame, 
        generated_dataset_base: pandas.DataFrame,
        generated_dataset_fine_tuned: pandas.DataFrame,
        random_seed = 42):
    """
    Function that calculates and prints the confidence Intervals.

    Parameters
    -----------
    original_dataset: the dataset containing the gold answers.
    generated_dataset_base: the dataset containing the base model's generated answers.
    generated_dataset_fine_tuned: the dataset containing the fine_tuned model's generated answers.

    Returns
    --------
    None.
    """
    # Set random seed for reproducibility.
    numpy.random.seed(random_seed)

    y_base = numpy.array([
        extract_letter_from_generated_answer(answer, options)
            for answer, options in zip(
                generated_dataset_base['generated_answer'], 
                original_dataset['answers']
            )
    ])
        
    y_fine_tuned = numpy.array([
        extract_letter_from_generated_answer(answer, options)
            for answer, options in zip(
                generated_dataset_fine_tuned['generated_answer'], 
                original_dataset['answers']
            )
    ])

    y_true = numpy.array(generated_dataset_base['correct_answer'].to_list())
    
    n_iterations = 10000
    bootstrap_diffs = []
    n = len(y_true)
    
    # Calculate the accuracies using sklearn.
    acc_base_true = accuracy_score(y_true, y_base)
    acc_fine_tuned_true = accuracy_score(y_true, y_fine_tuned)
    exact_diff = acc_fine_tuned_true - acc_base_true
    
    # Calculate binary 'correct' arrays for faster bootstrapping.
    base_correct = (y_base == y_true).astype(float)
    fine_tuned_correct = (y_fine_tuned == y_true).astype(float)
     
    # Run bootstrap strictly for the Confidence Interval.
    for _ in range(n_iterations):
        # Sample with replacement.
        indices = numpy.random.choice(numpy.arange(n), size = n, replace = True)
        
        acc_base_boot = numpy.mean(base_correct[indices])
        acc_ft_boot = numpy.mean(fine_tuned_correct[indices])
        
        bootstrap_diffs.append(acc_ft_boot - acc_base_boot)
    
    # Calculate 95% Confidence Interval.
    lower = numpy.percentile(bootstrap_diffs, 2.5)
    upper = numpy.percentile(bootstrap_diffs, 97.5)
    
    # Print results.
    print(f'Base Accuracy: {acc_base_true * 100:.2f}%')
    print(f'Fine-Tuning Accuracy: {acc_fine_tuned_true * 100:.2f}%')
    print(f'Exact Improvement: {exact_diff * 100:.2f}%')
    print(f'95% CI: [{lower * 100:.2f}%, {upper * 100:.2f}%]')
    
    if lower > 0:
        print('Significant: The entire confidence interval is above zero.')
    else:
        print('Not significant: The interval includes zero or negative values.')

    return


def calculate_bootstrap_bertscore(
        f1_base_scores: numpy.typing.NDArray,
        f1_fine_tuned_scores: numpy.typing.NDArray,
        random_seed: int = 42):
    """
    Performs bootstrap resampling to test if the fine-tuned model is 
    significantly better than the base model on open-ended QA tasks
    using BERTScore.

    Parameters
    -----------    
    f1_base_scores: the numpy array containing the f1 scores for the base model.
    f1_fine_tuned_scores: the numpy array containing the f1 scores for the generated model.
    
    Returns
    --------
    None.
    """
    # Set random seed for reproducibility.
    numpy.random.seed(random_seed)

    # Calculate the actual observed difference in means.
    mean_base = numpy.mean(f1_base_scores)
    mean_ft = numpy.mean(f1_fine_tuned_scores)
    exact_diff = mean_ft - mean_base

    # Bootstrap Resampling.
    bootstrap_differences = []
    n_iterations = 10000
    n = len(f1_base_scores)

    for _ in range(n_iterations):
        # Sample indices with replacement.
        indices = numpy.random.choice(n, size = n, replace = True)
        
        # Calculate means on this resampled dataset.
        boot_mean_base = numpy.mean(f1_base_scores[indices])
        boot_mean_ft = numpy.mean(f1_fine_tuned_scores[indices])
        
        # Track the difference for the confidence interval.
        boot_diff = boot_mean_ft - boot_mean_base
        bootstrap_differences.append(boot_diff)
    
    # Calculate 95% Confidence Interval for the difference in means.
    lower = numpy.percentile(bootstrap_differences, 2.5)
    upper = numpy.percentile(bootstrap_differences, 97.5)

    # Print results.
    print(f'Base BERTScore: {mean_base * 100:.2f}%')
    print(f'Fine-Tuning BERTScore: {mean_ft * 100:.2f}%')
    print(f'Exact Improvement: {exact_diff * 100:.2f}%')
    print(f'95% CI: [{lower * 100:.2f}%, {upper * 100:.2f}%]')
    
    if lower > 0:
        print('Significant: The entire confidence interval is above zero.')
    else:
        print('Not significant: The interval includes zero or negative values.')

    return
