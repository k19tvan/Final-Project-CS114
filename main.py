import argparse
import sys
from src.experiment1 import run_experiment_1
from src.experiment2 import run_experiment_2
from src.experiment3 import run_experiment_3

def main():
    parser = argparse.ArgumentParser(description="Machine Learning Project - Image Classification Experiments")
    parser.add_argument("--exp", type=int, choices=[1, 2, 3], help="Experiment number to run (1, 2, or 3)")
    parser.add_argument("--all", action="store_true", help="Run all experiments sequentially")
    
    args = parser.parse_args()
    
    if args.all:
        run_experiment_1()
        run_experiment_2()
        run_experiment_3()
    elif args.exp == 1:
        run_experiment_1()
    elif args.exp == 2:
        run_experiment_2()
    elif args.exp == 3:
        run_experiment_3()
    else:
        print("Please specify an experiment to run using --exp [1|2|3] or --all.")
        parser.print_help()

if __name__ == "__main__":
    main()
