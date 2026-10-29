import pandas as pd
import numpy as np


from benchmark_upstream_heads import benchmark_heads, risk_coverage_curve_regression
from benchmark_upstream_finetuned import make_upstream_results


import matplotlib.pyplot as plt


def plot_risk_coverage_facets(
    results,
    coverage=None,
    figsize=(12, 4),
    save_path="risk_coverage_finetuning_facets",
):
    """
    Plot risk-coverage curves in 3 separate panels:
    one for each fine-tuning regime (0%, 25%, 100%).

    Parameters
    ----------
    results : dict
        Dictionary like:
        {
            "gaussian_0": ...,
            "student_t_0": ...,
            "mc_dropout_0": ...,
            "gaussian_25": ...,
            ...
        }

    coverage : array-like or None
        Coverage values. If None, uses [100, 90, ..., 10].

    figsize : tuple
        Figure size.

    save_path : str or None
        Base file name to save .pdf and .png.
    """

    if coverage is None:
        coverage = np.arange(100, 0, -10)

    coverage = np.asarray(coverage)

    model_styles = {
        "gaussian": {
            "label": "Gaussian",
            "marker": "o",
            "linestyle": "-",
        },
        "student_t": {
            "label": r"Student-$t$",
            "marker": "s",
            "linestyle": "-",
        },
        "mc_dropout": {
            "label": "MC Dropout",
            "marker": "^",
            "linestyle": "-",
        },
    }

    regimes = [0, 25, 100]

    fig, axes = plt.subplots(
        1, 3,
        figsize=figsize,
        sharey=True
    )

    for ax, regime in zip(axes, regimes):
        for model, style in model_styles.items():
            key = f"{model}_{regime}"

            if key not in results:
                continue

            values = np.asarray(results[key])

            if len(values) != len(coverage):
                raise ValueError(
                    f"{key}: expected {len(coverage)} values, got {len(values)}"
                )

            ax.plot(
                coverage,
                values,
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.0,
                markersize=5,
                label=style["label"],
            )

        ax.set_title(f"{regime}% fine-tuning")
        ax.set_xlabel("Coverage (%)")
        ax.set_xticks(coverage)
        ax.invert_xaxis()
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    axes[0].set_ylabel("MSE")

    # single legend for the whole figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.05),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is not None:
        fig.savefig(f"{save_path}.pdf", bbox_inches="tight")
        fig.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")

    plt.show()

    return fig, axes

class UpstreamResults:
    
    def __init__(self, root_path = 'ssl_wearables/output_hr/'):
        
        self.head_types = ['mc_dropout','gaussian', 'student_t']
        
        self.output_sizes = {head: i+1 for i, head in enumerate(self.head_types)}
        
        self.finetuned_encoders = self.get_paths(root_path)
        self.results = {}
        
    def get_paths(self, root_path):
        
        paths = {'mc_dropout': {25: root_path+ 'HR_reg_mc_dropout_oxfordhub_ft25_25_final.mdl', 100:root_path+ 'HR_reg_mc_dropout_oxfordhub_ft100_100_final.mdl'},
                'gaussian':{25:root_path + 'HR_reg_gaussian_oxfordhub_25_final.mdl', 100:root_path + 'HR_reg_gaussian_oxfordhub_100_final.mdl'},
                'student_t':{25:root_path + 'HR_reg_student_t_oxfordhub_25_final.mdl', 100:root_path + 'HR_reg_student_t_oxfordhub_100_final.mdl'}}
        return paths
    
    def get_0_finetuned_results(self):
    
        df = benchmark_heads()
        results_0 = {}

        for head_type in self.head_types:

            df_head = df[df.head_name == head_type]
            df_head.var = df_head["var"]
            df_summary = risk_coverage_curve_regression(df_head.y.values, df_head.mu.values, df_head.var.values, df_head.raw_hr.values)

            results_0[head_type + '_0'] = df_summary.mse.values

        self.results.update(results_0)
    
    def get_25_100_finetuned_results(self):
        
        results_finetuned = {}
        
        for head_name, path_dict in self.finetuned_encoders.items():
            
            output_size = self.output_sizes[head_name]
            
            for finetune_deepth, encoder_path in path_dict.items():

                df = make_upstream_results(output_size, encoder_path)

                df_summary = risk_coverage_curve_regression(df.y.values, df.predicted_hr.values, df.hr_uncertainty.values,df.original.values)

                results_finetuned[head_name + f"_{finetune_deepth}"] = df_summary.mse.values
        
        self.results.update(results_finetuned)
    
    def make_comparison_plot(self):
        
        plot_risk_coverage_facets(self.results)
    
def main():
    
    results = UpstreamResults()
    
    results.get_0_finetuned_results()
    results.get_25_100_finetuned_results()
    
    data = {'method': [], 'delta_MSE': [], 'initial_MSE':[]}

    for method, result in results.results.items():

        initial_mse = result[0]
        delta = (result[-1]- initial_mse)/initial_mse

        data['method'].append(method)
        data['delta_MSE'].append(delta)
        data['initial_MSE'].append(initial_mse)
        
    df = pd.DataFrame(data=data)
    df.to_csv("delta_upstream_results.csv")
    
    results.make_comparison_plot()

if __name__ == "__main__":
    main()