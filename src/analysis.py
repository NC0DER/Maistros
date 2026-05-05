import os
import json
import numpy
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches


from tqdm import tqdm
from src.utils import *
from src.config import *


def load_datasets() -> list[pandas.DataFrame]:
    dataframes_list = []

    # Get the two different types of datasets.
    datasets = datasets_mc + datasets_oe

    for dataset_name in datasets:
        file_path = os.path.join(project_dir_path,'datasets', f'{dataset_name}.csv')
        dataframes_list.append(pandas.read_csv(file_path, index_col = False))
    return dataframes_list


def get_stats_dict(label, data, category):
    p5, p25, p50, p75, p95, p99 = numpy.percentile(data, [5, 25, 50, 75, 95, 99])
    
    return {
        'Dataset': label,
        'Type': category,
        'Min': numpy.min(data),
        'P5': round(p5, 1),
        'P25 (Q1)': p25,
        'P50 (Median)': p50,
        'Mean': round(numpy.mean(data), 2),
        'P75 (Q3)': p75,
        'P95': p95,
        'P99': p99,
        'Max': numpy.max(data),
        'IQR': p75 - p25
    }


def analyze():
    dfs = load_datasets()
    question_length_lists = []
    best_answer_length_lists = []
    all_stats_records = []

    # Helper to get ID column or fallback to index.
    def get_sample_id(df, idx):
        return df.loc[idx, 'id'] if 'id' in df.columns else f'Row {idx}'

    print('\nAnalyzing Datasets & Identifying Outliers...\n')

    labels = [
        'GPCR', 'INCLUDE', 'MCQA Greek ASEP', 
        'Greek Medical MCQA', 'Plutus QA', 
        'Truthful QA Greek', 'GreekMMLU', 
        'DemosQA', 'Greek Civics QA', 'CulturaQA'
    ]

    for label, df in tqdm(zip(labels, dfs), total = len(labels)):
        
        # Measure lengths in tokens.
        current_q_lengths = measure_token_lengths(df, 'question')

        if label not in ['CulturaQA']:
            current_a_lengths = measure_token_lengths(df, 'best_answer')
        else:
            current_a_lengths = measure_token_lengths(df, 'answer')

        # Attach lengths to the dataframe temporarily for ID lookup.
        df['temp_q_len'] = current_q_lengths
        df['temp_a_len'] = current_a_lengths

        # Find outliers of length 1.
        len1_qs = df[df['temp_q_len'] == 1]
        len1_as = df[df['temp_a_len'] == 1]

        # Find max sample IDs.
        max_q_idx = df['temp_q_len'].idxmax()
        max_a_idx = df['temp_a_len'].idxmax()

        # Print highlights.
        print(f'\n[{label}] Highlights:')
        print(f'  - Max Question Length: {df['temp_q_len'].max()} (ID: {get_sample_id(df, max_q_idx)})')
        print(f'  - Max Answer Length:   {df['temp_a_len'].max()} (ID: {get_sample_id(df, max_a_idx)})')
        
        if not len1_qs.empty:
            print(f'  - Samples with Q-Length 1 (IDs): {df.loc[len1_qs.index[:5]].apply(lambda x: get_sample_id(df, x.name), axis = 1).tolist()}')
        if not len1_as.empty:
            print(f'  - Samples with A-Length 1 (IDs): {df.loc[len1_as.index[:5]].apply(lambda x: get_sample_id(df, x.name), axis = 1).tolist()}')

        # Record statistics.
        all_stats_records.append(get_stats_dict(label, current_q_lengths, 'Question'))
        all_stats_records.append(get_stats_dict(label, current_a_lengths, 'Answer'))

        question_length_lists.append(current_q_lengths)
        best_answer_length_lists.append(current_a_lengths)

    # Save the dataframe to a .csv.
    stats_df = pandas.DataFrame(all_stats_records)
    save_dir = os.path.join(project_dir_path, 'dataset_length_statistics.csv')
    stats_df.to_csv(save_dir, index = False)

    return


def plot_train_val_losses():

    # Set the path for the trainer state.json and read this file.
    total_checkpoints = 500
    json_path = os.path.join(
        project_dir_path, 
        'bs16_rank32', 
        f'checkpoint-{total_checkpoints}', 
        'trainer_state.json'
    )
    with open(json_path, 'r') as file:
        data = json.load(file)

    # Initialize lists/dicts to store steps and losses.
    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []
    train_dict = {} # Used for quick lookup between steps and losses to draw the box.

    # Find the first eval step to filter training data.
    first_eval_step = min([entry['step'] for entry in data['log_history'] if 'eval_loss' in entry])

    # Iterate through the log history to extract losses.
    for entry in data['log_history']:
        step = entry['step']
        if 'loss' in entry:
            # Store all training losses for each step.
            train_dict[step] = entry['loss'] 

            # Skip training losses before the first eval step.
            if step < first_eval_step:
                continue

            train_steps.append(step)
            train_losses.append(entry['loss'])
            
        if 'eval_loss' in entry:
            eval_steps.append(step)
            eval_losses.append(entry['eval_loss'])

    # Find the best model point, where the smallest validation loss is achieved. 
    min_eval_loss = min(eval_losses)
    best_step = eval_steps[eval_losses.index(min_eval_loss)]
    best_train_loss = train_dict.get(best_step)

    # Create the figure and primary axis.
    plt.rcParams.update({
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'axes.edgecolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'legend.edgecolor': 'black'
    })
    _, ax1 = plt.subplots(figsize = (10, 6))
    
    # Plot training and validation losses on the primary axis.
    ax1.plot(train_steps, train_losses, label = 'Training Loss', color = 'blue', marker = 'o', alpha = 0.7)
    ax1.plot(eval_steps, eval_losses, label = 'Validation Loss', color = 'red', marker = 's')

    # Add the Visual Marker.
    if best_train_loss is not None:
        # Define box dimensions
        width = 10  # Width in steps.
        padding_y = 0.015 
        
        y_bottom = min(min_eval_loss, best_train_loss) - padding_y
        y_top = max(min_eval_loss, best_train_loss) + padding_y
        height = y_top - y_bottom
        
        # Create and add the rectangle.
        rect = patches.Rectangle(
            (best_step - (width / 2), y_bottom), 
            width, height,
            linewidth = 2, edgecolor = 'red', facecolor = 'red', alpha = 0.1,
            label = 'Best Model Point'
        )
        ax1.add_patch(rect)
        
        # Add text annotation.
        ax1.text(
            best_step, y_top + 0.005, f'Best Step: {best_step}', 
            color = 'red', fontweight = 'bold', ha = 'center', fontsize = 9
        )

    # Add labels to the primary axis.
    ax1.set_xlabel('Global Steps')
    ax1.set_ylabel('Loss')
    ax1.legend(loc = 'upper right')
    ax1.grid(True, linestyle = '--', alpha = 0.7)

    # Create a secondary X-axis for epochs.
    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())

    # Define the custom epoch ticks.
    epoch_step_locations = [(i * total_checkpoints) // 4 for i in range(1, 5)]
    epoch_string_labels = ['Epoch 1', 'Epoch 2', 'Epoch 3', 'Epoch 4']
    
    ax2.set_xticks(epoch_step_locations)
    ax2.set_xticklabels(epoch_string_labels)

    # Save the plot as a high dpi png.
    save_dir = os.path.join(project_dir_path, 'images', 'train_val_loss_plot.png')
    plt.tight_layout()
    plt.savefig(save_dir, dpi = 300)

    return

def plot_category_distribution():

    # Define the path for the dataset and load it.
    dataset_path = os.path.join(
        project_dir_path, 
        'all_data.csv'
    )
    df = pandas.read_csv(dataset_path)

    # Set up the plot style.
    plt.figure(figsize = (10, 6))
    sns.set_theme(
        style = 'whitegrid',
        rc = {
            'text.color': 'black',
            'axes.labelcolor': 'black',
            'axes.edgecolor': 'black',
            'xtick.color': 'black',
            'ytick.color': 'black'
        }
    )

    # Translate the Greek labels to English.
    translations = {
        'πολιτισμός': 'culture',
        'ταξίδια': 'travelling',
        'επιστήμη': 'science',
        'ιστορία': 'history',
        'οικονομία': 'economy',
        'υγεία': 'health',
        'εκπαίδευση': 'education',
        'δίκαιο': 'law',
        'αθλητισμός': 'sports',
        'πολιτική': 'politics',
        'φαγητό': 'food'
    }

    # Update the dataframe labels to have categories translated.
    df['category'] = df['category'].apply(
        lambda x: f'{x}\n({translations[x]})'
    )

    # Create a categorical distribution plot where elements are sorted by frequency.
    order = df['category'].value_counts().index
    ax = sns.countplot(
        data = df, 
        x = 'category', 
        order = order, 
        color = 'royalblue'
    )

    # Add percentage annotations on top of each bar.
    total_samples = len(df)
    for p in ax.patches:
        # Calculate the percentage.
        percentage = f'{100 * p.get_height() / total_samples:.1f}%'
        
        # Determine the position (center of the bar, slightly above the top).
        x = p.get_x() + p.get_width() / 2
        y = p.get_height() + (total_samples * 0.001) # Small offset above bar
        
        ax.annotate(
            percentage, 
            (x, y), 
            ha = 'center', 
            va = 'bottom', 
            fontsize = 12, 
            fontweight = 'bold'
        )

    # Set title and labels.
    plt.title(f'Cultura QA - Category Distribution', fontsize = 16)
    plt.ylabel('# Samples', fontsize = 14)
    plt.xlabel(None)
    plt.xticks(rotation = 0, fontsize = 11)

    # Increase the y-axis limit slightly to make room for the labels above the bars.
    plt.ylim(0, df['category'].value_counts().max() * 1.1)
    
    # Ensure layout fits well without cutting off labels.
    plt.tight_layout()

    # Save the plot as a high dpi png.
    save_dir = os.path.join(project_dir_path, 'images', 'category_distribution_plot.png')
    plt.savefig(save_dir, dpi = 300)

    return
