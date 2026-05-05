from src.dataset import (
    create_synthetic_data,
    process_data_and_form_csv,
    postprocess,
    tokenizer_count
)
from src.training import (
    fine_tune_conversational_QA,
    merge_adapter_weights_to_base_and_save,
    quantize_and_save_model,
    upload_model,
    upload_quantized_model
)
from src.process import process_all
from src.analysis import (
    analyze,
    plot_train_val_losses, 
    plot_category_distribution
)
from src.evaluate import (
    run_mcqa_experiments,
    run_bertscore_experiments,
    run_statistical_significance_tests
)

def create_synthetic_dataset():
    create_synthetic_data()
    process_data_and_form_csv()
    postprocess()
    tokenizer_count()

def save_models():
    merge_adapter_weights_to_base_and_save()
    quantize_and_save_model(model_type = 'merged')
    quantize_and_save_model(model_type = 'base')
    quantize_and_save_model(model_type = 'EuroLLM')

def upload_models():
    upload_model('Maistros-8B-Instruct')
    upload_quantized_model('Maistros-8B-Instruct-4bit')


def main():
    create_synthetic_dataset()
    fine_tune_conversational_QA()
    save_models()
    process_all()
    analyze()
    plot_train_val_losses()
    plot_category_distribution()
    run_mcqa_experiments()
    run_bertscore_experiments()
    run_statistical_significance_tests()
    upload_models()

    return

if __name__ == '__main__': main()
