from src.evaluate import (
    run_mcqa_experiments,
    run_bertscore_experiments,
    run_statistical_significance_tests
)

def reproduce_experiments():
    run_mcqa_experiments()
    run_bertscore_experiments()
    run_statistical_significance_tests()

    return

if __name__ == '__main__': reproduce_experiments()
