import json
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class HyperparameterAnalyzer:
    """Analyzes hyperparameter experiment results and generates visualizations."""

    def __init__(self, results_dir: str, output_dir: str = "analysis_results"):
        """
        Initialize the analyzer.

        Args:
            results_dir: Directory containing JSON result files
            output_dir: Directory to save analysis plots and reports
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = []
        self.df = None

    def load_results(self) -> None:
        """Load all JSON result files from the results directory."""
        json_files = sorted(self.results_dir.glob("study_*.json"))

        if not json_files:
            raise FileNotFoundError(
                f"No JSON files found in {self.results_dir}")

        print(f"Found {len(json_files)} result files")

        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
                # Extract study and trial numbers from filename
                parts = json_file.stem.split('_')
                study_num = int(parts[1])
                trial_num = int(parts[3])

                data['study_num'] = study_num
                data['trial_num'] = trial_num
                self.results.append(data)

        print(f"Loaded {len(self.results)} experiments")
        self._create_dataframe()

    def _create_dataframe(self) -> None:
        """Convert loaded results into a pandas DataFrame for analysis."""
        rows = []

        for result in self.results:
            row = {
                'study_num': result['study_num'],
                'trial_num': result['trial_num'],
                'best_val_acc5': result['best_val_acc5'],
            }

            # Add all parameters
            for param, value in result['params'].items():
                row[param] = value

            # Add final epoch metrics
            if result['history']['val_acc5']:
                row['final_val_acc5'] = result['history']['val_acc5'][-1]
                row['final_val_ppl'] = result['history']['val_ppl'][-1]
                row['final_val_loss'] = result['history']['val_loss'][-1]
                row['final_train_loss'] = result['history']['train_loss'][-1]

            # Add trial parameters info
            row['trial_params'] = ','.join(result['trial_params'])

            rows.append(row)

        self.df = pd.DataFrame(rows)
        print(f"\nDataFrame shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")

    def print_summary_statistics(self) -> None:
        """Print summary statistics of the experiments."""
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)

        print(
            f"\nBest validation accuracy (top-5): {self.df['best_val_acc5'].max():.4f}")
        print(
            f"Worst validation accuracy (top-5): {self.df['best_val_acc5'].min():.4f}")
        print(
            f"Mean validation accuracy (top-5): {self.df['best_val_acc5'].mean():.4f}")
        print(
            f"Std validation accuracy (top-5): {self.df['best_val_acc5'].std():.4f}")

        print("\n" + "-" * 80)
        print("Top 5 configurations:")
        print("-" * 80)
        top_configs = self.df.nlargest(5, 'best_val_acc5')

        for idx, row in top_configs.iterrows():
            print(f"\nRank {list(top_configs.index).index(idx) + 1}:")
            print(f"  Accuracy: {row['best_val_acc5']:.4f}")
            print(f"  Study: {row['study_num']}, Trial: {row['trial_num']}")
            print(f"  Config: {row['trial_params']}")

    def analyze_parameter_impact(self) -> pd.DataFrame:
        """
        Analyze the impact of each hyperparameter on performance.

        Returns:
            DataFrame with parameter impact statistics
        """
        print("\n" + "=" * 80)
        print("PARAMETER IMPACT ANALYSIS")
        print("=" * 80)

        # Get hyperparameter columns (exclude metadata and results)
        exclude_cols = ['study_num', 'trial_num', 'best_val_acc5',
                        'final_val_acc5',
                        'final_val_ppl', 'final_val_loss', 'final_train_loss',
                        'trial_params']
        param_cols = [col for col in self.df.columns if col not in exclude_cols]

        impact_data = []

        for param in param_cols:
            if self.df[param].nunique() <= 1:
                continue  # Skip parameters that don't vary

            # Group by parameter value and compute statistics
            grouped = self.df.groupby(param)['best_val_acc5'].agg(
                ['mean', 'std', 'count', 'min', 'max'])

            # Compute range (max - min of means across groups)
            impact_range = grouped['mean'].max() - grouped['mean'].min()

            # Compute coefficient of variation
            cv = self.df['best_val_acc5'].std() / self.df[
                'best_val_acc5'].mean()

            # Try ANOVA if numeric parameter with multiple values
            try:
                groups = [group['best_val_acc5'].values for name, group in
                          self.df.groupby(param)]
                if len(groups) > 1 and all(len(g) > 0 for g in groups):
                    f_stat, p_value = stats.f_oneway(*groups)
                else:
                    f_stat, p_value = np.nan, np.nan
            except:
                f_stat, p_value = np.nan, np.nan

            impact_data.append({
                'parameter': param,
                'n_unique_values': self.df[param].nunique(),
                'impact_range': impact_range,
                'relative_impact': impact_range / self.df[
                    'best_val_acc5'].mean() * 100,
                'f_statistic': f_stat,
                'p_value': p_value,
                'mean_acc_std': grouped['mean'].std(),
            })

        impact_df = pd.DataFrame(impact_data)
        impact_df = impact_df.sort_values('impact_range', ascending=False)

        print("\nParameter Impact Ranking (by range of mean accuracy):")
        print("-" * 80)
        for idx, row in impact_df.iterrows():
            print(f"{row['parameter']:20s}: "
                  f"Range={row['impact_range']:.4f} "
                  f"({row['relative_impact']:.2f}%), "
                  f"F-stat={row['f_statistic']:.2f}, "
                  f"p-value={row['p_value']:.4f}")

        # Save to CSV
        impact_df.to_csv(self.output_dir / "parameter_impact_analysis.csv",
                         index=False)

        return impact_df

    def plot_parameter_distributions(self) -> None:
        """Plot distribution of accuracy for each hyperparameter value."""
        exclude_cols = ['study_num', 'trial_num', 'best_val_acc5',
                        'final_val_acc5',
                        'final_val_ppl', 'final_val_loss', 'final_train_loss',
                        'trial_params']
        param_cols = [col for col in self.df.columns if col not in exclude_cols]

        # Filter parameters that vary
        varying_params = [col for col in param_cols if
                          self.df[col].nunique() > 1]

        n_params = len(varying_params)
        n_cols = 3
        n_rows = (n_params + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        for idx, param in enumerate(varying_params):
            ax = axes[idx]

            # Sort by parameter value for better visualization
            grouped = self.df.groupby(param)['best_val_acc5'].apply(list)
            sorted_keys = sorted(grouped.keys(),
                                 key=lambda x: (isinstance(x, str), x))

            data_to_plot = [grouped[key] for key in sorted_keys]
            labels = [str(key) for key in sorted_keys]

            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)

            # Color boxes
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')

            ax.set_xlabel(param, fontsize=12, fontweight='bold')
            ax.set_ylabel('Best Val Accuracy (Top-5)', fontsize=10)
            ax.set_title(f'Impact of {param}', fontsize=12)
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

        # Hide unused subplots
        for idx in range(n_params, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / "parameter_distributions.png", dpi=300,
                    bbox_inches='tight')
        print(f"\nSaved: parameter_distributions.png")
        plt.close()

    def plot_correlation_heatmap(self) -> None:
        """Plot correlation heatmap between hyperparameters and performance."""
        # Select numeric columns only
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        exclude = ['study_num', 'trial_num']
        numeric_cols = [col for col in numeric_cols if col not in exclude]

        if len(numeric_cols) < 2:
            print("Not enough numeric columns for correlation analysis")
            return

        # Compute correlation matrix
        corr_matrix = self.df[numeric_cols].corr()

        # Focus on correlations with best_val_acc5
        if 'best_val_acc5' in corr_matrix.columns:
            acc_corr = corr_matrix['best_val_acc5'].sort_values(ascending=False)

            print("\n" + "=" * 80)
            print("CORRELATION WITH VALIDATION ACCURACY")
            print("=" * 80)
            for param, corr in acc_corr.items():
                if param != 'best_val_acc5':
                    print(f"{param:20s}: {corr:7.4f}")

        # Plot full correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                    center=0, square=True, linewidths=1)
        plt.title('Correlation Heatmap of Hyperparameters and Metrics',
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "correlation_heatmap.png", dpi=300,
                    bbox_inches='tight')
        print(f"\nSaved: correlation_heatmap.png")
        plt.close()

    def plot_study_progression(self) -> None:
        """Plot how performance evolved across studies."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Best accuracy per study
        ax1 = axes[0]
        study_best = self.df.groupby('study_num')['best_val_acc5'].max()
        study_mean = self.df.groupby('study_num')['best_val_acc5'].mean()

        x = study_best.index
        ax1.plot(x, study_best.values, 'o-', label='Best in study', linewidth=2,
                 markersize=8)
        ax1.plot(x, study_mean.values, 's--', label='Mean in study',
                 linewidth=2, markersize=8)
        ax1.set_xlabel('Study Number', fontsize=12)
        ax1.set_ylabel('Validation Accuracy (Top-5)', fontsize=12)
        ax1.set_title('Grid Search Progression Across Studies', fontsize=14,
                      fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Plot 2: Distribution per study
        ax2 = axes[1]
        study_data = [self.df[self.df['study_num'] == s]['best_val_acc5'].values
                      for s in sorted(self.df['study_num'].unique())]
        bp = ax2.boxplot(study_data,
                         labels=[f'Study {i}' for i in range(len(study_data))],
                         patch_artist=True)

        for patch in bp['boxes']:
            patch.set_facecolor('lightgreen')

        ax2.set_xlabel('Study', fontsize=12)
        ax2.set_ylabel('Validation Accuracy (Top-5)', fontsize=12)
        ax2.set_title('Accuracy Distribution per Study', fontsize=14,
                      fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "study_progression.png", dpi=300,
                    bbox_inches='tight')
        print(f"\nSaved: study_progression.png")
        plt.close()

    def plot_training_curves(self, top_n: int = 5) -> None:
        """Plot training curves for top N configurations."""
        top_configs = self.df.nlargest(top_n, 'best_val_acc5')

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        for idx, (_, row) in enumerate(top_configs.iterrows()):
            # Find the original result to get history
            result = next(r for r in self.results
                          if r['study_num'] == row['study_num']
                          and r['trial_num'] == row['trial_num'])

            history = result['history']
            epochs = range(1, len(history['train_loss']) + 1)

            label = f"Study {row['study_num']}, Trial {row['trial_num']} (acc={row['best_val_acc5']:.4f})"

            # Training loss
            axes[0, 0].plot(epochs, history['train_loss'], marker='o',
                            label=label)

            # Validation loss
            axes[0, 1].plot(epochs, history['val_loss'], marker='o',
                            label=label)

            # Validation perplexity
            axes[1, 0].plot(epochs, history['val_ppl'], marker='o', label=label)

            # Validation accuracy
            axes[1, 1].plot(epochs, history['val_acc5'], marker='o',
                            label=label)

        axes[0, 0].set_title('Training Loss', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(alpha=0.3)

        axes[0, 1].set_title('Validation Loss', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(alpha=0.3)

        axes[1, 0].set_title('Validation Perplexity', fontsize=12,
                             fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Perplexity')
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(alpha=0.3)

        axes[1, 1].set_title('Validation Accuracy (Top-5)', fontsize=12,
                             fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(alpha=0.3)

        plt.suptitle(f'Training Curves for Top {top_n} Configurations',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_curves_top_configs.png",
                    dpi=300, bbox_inches='tight')
        print(f"\nSaved: training_curves_top_configs.png")
        plt.close()

    def plot_parameter_pairs(self, top_params: List[str]) -> None:
        """Plot 2D scatter plots for pairs of most important parameters."""
        if len(top_params) < 2:
            print("Not enough parameters for pair plotting")
            return

        # Take top 4 parameters for pair plots
        params_to_plot = top_params[:min(4, len(top_params))]

        n_params = len(params_to_plot)

        fig, axes = plt.subplots(n_params, n_params, figsize=(15, 15))

        if n_params == 1:
            axes = np.array([[axes]])
        elif axes.ndim == 1:
            axes = axes.reshape(n_params, 1)

        for i, param1 in enumerate(params_to_plot):
            for j, param2 in enumerate(params_to_plot):
                ax = axes[i, j]

                if i == j:
                    # Diagonal: histogram
                    ax.hist(self.df[param1],
                            bins=min(20, self.df[param1].nunique()),
                            alpha=0.7, color='steelblue', edgecolor='black')
                    ax.set_ylabel('Count')
                else:
                    # Off-diagonal: scatter plot
                    scatter = ax.scatter(self.df[param2], self.df[param1],
                                         c=self.df['best_val_acc5'],
                                         cmap='viridis', s=100, alpha=0.6)

                    if j == n_params - 1 and i == 0:
                        cbar = plt.colorbar(scatter, ax=ax)
                        cbar.set_label('Best Val Acc5', rotation=270,
                                       labelpad=15)

                if i == n_params - 1:
                    ax.set_xlabel(param2, fontsize=10)
                else:
                    ax.set_xticklabels([])

                if j == 0 and i > 0:
                    ax.set_ylabel(param1, fontsize=10)
                elif j > 0:
                    ax.set_yticklabels([])

        plt.suptitle('Parameter Pair Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "parameter_pairs.png", dpi=300,
                    bbox_inches='tight')
        print(f"\nSaved: parameter_pairs.png")
        plt.close()

    def generate_report(self) -> None:
        """Generate a comprehensive text report of the analysis."""
        report_path = self.output_dir / "analysis_report.txt"

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("HYPERPARAMETER ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Total experiments analyzed: {len(self.df)}\n")
            f.write(f"Number of studies: {self.df['study_num'].nunique()}\n\n")

            f.write("-" * 80 + "\n")
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(
                f"Best validation accuracy: {self.df['best_val_acc5'].max():.6f}\n")
            f.write(
                f"Worst validation accuracy: {self.df['best_val_acc5'].min():.6f}\n")
            f.write(
                f"Mean validation accuracy: {self.df['best_val_acc5'].mean():.6f}\n")
            f.write(
                f"Std validation accuracy: {self.df['best_val_acc5'].std():.6f}\n")
            f.write(
                f"Coefficient of variation: {self.df['best_val_acc5'].std() / self.df['best_val_acc5'].mean() * 100:.2f}%\n\n")

            f.write("-" * 80 + "\n")
            f.write("BEST CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            best_row = self.df.loc[self.df['best_val_acc5'].idxmax()]
            f.write(
                f"Study {best_row['study_num']}, Trial {best_row['trial_num']}\n")
            f.write(f"Validation Accuracy: {best_row['best_val_acc5']:.6f}\n\n")

            exclude_cols = ['study_num', 'trial_num', 'best_val_acc5',
                            'final_val_acc5',
                            'final_val_ppl', 'final_val_loss',
                            'final_train_loss', 'trial_params']
            param_cols = [col for col in self.df.columns if
                          col not in exclude_cols]

            f.write("Parameters:\n")
            for param in param_cols:
                f.write(f"  {param}: {best_row[param]}\n")

            f.write("\n" + "-" * 80 + "\n")
            f.write("FILES GENERATED\n")
            f.write("-" * 80 + "\n")
            f.write("- parameter_impact_analysis.csv\n")
            f.write("- parameter_distributions.png\n")
            f.write("- correlation_heatmap.png\n")
            f.write("- study_progression.png\n")
            f.write("- training_curves_top_configs.png\n")
            f.write("- parameter_pairs.png\n")
            f.write("- analysis_report.txt\n")

        print(f"\nSaved: analysis_report.txt")

    def run_full_analysis(self) -> None:
        """Run complete analysis pipeline."""
        print("\n" + "=" * 80)
        print("RUNNING FULL HYPERPARAMETER ANALYSIS")
        print("=" * 80)

        self.load_results()
        self.print_summary_statistics()

        impact_df = self.analyze_parameter_impact()
        top_params = impact_df['parameter'].head(4).tolist()

        self.plot_parameter_distributions()
        self.plot_correlation_heatmap()
        self.plot_study_progression()
        self.plot_training_curves(top_n=5)
        self.plot_parameter_pairs(top_params)
        self.generate_report()

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"All results saved to: {self.output_dir}")


def main():
    """Main execution function."""
    # Configuration
    RESULTS_DIR = "next_token_prediction_model/study_results"
    OUTPUT_DIR = "hyperparameter_analysis"

    # Create analyzer
    analyzer = HyperparameterAnalyzer(
        results_dir=RESULTS_DIR,
        output_dir=OUTPUT_DIR
    )

    # Run full analysis
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()