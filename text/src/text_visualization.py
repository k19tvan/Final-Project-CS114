import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.1)

def plot_with_std(results, x_values, xlabel, title, save_path=None):
    plt.figure(figsize=(10, 6))
    
    model_colors = {
        'Multinomial NB (GLA)': '#2ca02c', 
        'Bernoulli NB (GLA)': '#ff7f0e',   
        'Linear SVM (DLA)': '#1f77b4',     
        'Logistic Regression (DLA)': '#d62728' 
    }
    default_colors = ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    for i, (name, (means, stds)) in enumerate(results.items()):
        means = np.array(means)
        stds = np.array(stds)
        c = model_colors.get(name, default_colors[i % len(default_colors)])
        
        plt.plot(x_values, means, label=name, color=c, marker='o', linewidth=2)
        plt.fill_between(x_values, means - stds, means + stds, color=c, alpha=0.15)
    
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel(xlabel)
    plt.ylabel('Accuracy (Mean ± Std)')
    plt.legend(frameon=True, loc='best')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    -> Plot saved: {save_path}")
        
    plt.close()