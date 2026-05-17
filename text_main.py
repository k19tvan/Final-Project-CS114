import warnings
import numpy as np
from src.text_data_loader import load_and_preprocess_data
from src.text_models import get_selected_models
from src.experiments.text_exp1_size import run_exp_size
from src.experiments.text_exp2_noise import run_exp_noise
from src.experiments.text_exp3_missing import run_exp_missing

warnings.filterwarnings('ignore')


GLOBAL_CONFIG = {
    "RANDOM_SEED": 24520130,        
    "PLOT_DIR": "plots/",        
    
    "N_SAMPLES": 3000,         
    "MAX_FEATURES": 3000,        
    
    "NB_ALPHA": 0.1,             #  Laplace Smoothing Naive Bayes (GLA)
    "MAX_ITER": 1000,           
    
    "CV_SPLITS": 10,              
    "EXP_SUBSET_SIZE": 1000,     
    
    "EXP1_SIZES": [50, 200, 500, 1000, 2000],             
    "EXP2_LABEL_NOISE": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],  
    "EXP2_FEATURE_NOISE": [0.0, 0.2, 0.5, 0.8, 1.0],      
    "EXP3_MISSING_RATES": [0.0, 0.2, 0.4, 0.6, 0.8]       
}

def main():
    np.random.seed(GLOBAL_CONFIG["RANDOM_SEED"])

    print("CS114 PROJECT: DLA vs GLA EVALUATION PIPELINE")

    X_tf_ag, X_bin_ag, y_ag = load_and_preprocess_data('ag_news', GLOBAL_CONFIG)
    X_tf_sg, X_bin_sg, y_sg = load_and_preprocess_data('sogou_news', GLOBAL_CONFIG)

    all_models = get_selected_models(GLOBAL_CONFIG) 

    print("EVALUATING: AG NEWS (ENGLISH)")
    run_exp_size(all_models, X_tf_ag, X_bin_ag, y_ag, "AG News", GLOBAL_CONFIG)
    run_exp_noise(all_models, X_tf_ag, X_bin_ag, y_ag, "AG News", GLOBAL_CONFIG)
    run_exp_missing(all_models, X_tf_ag, X_bin_ag, y_ag, "AG News", GLOBAL_CONFIG)

    print("EVALUATING: SOGOU NEWS (PINYIN)")
    run_exp_size(all_models, X_tf_sg, X_bin_sg, y_sg, "Sogou News", GLOBAL_CONFIG)
    run_exp_noise(all_models, X_tf_sg, X_bin_sg, y_sg, "Sogou News", GLOBAL_CONFIG)
    run_exp_missing(all_models, X_tf_sg, X_bin_sg, y_sg, "Sogou News", GLOBAL_CONFIG)
    
    print(f"[SUCCESS] All experiments finished. Check '{GLOBAL_CONFIG['PLOT_DIR']}' directory.")

if __name__ == "__main__":
    main()