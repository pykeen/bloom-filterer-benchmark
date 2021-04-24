import os
from typing import Optional

import humanize
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pykeen.datasets import get_dataset
from pykeen.sampling.filtering import BloomFilterer

HERE = os.path.abspath(os.path.dirname(__file__))
RESULTS_PATH = os.path.join(HERE, 'results.tsv')
PLOT_PATH = os.path.join(HERE, 'plot.svg')

DEFAULT_PRECISION = 5

#: Datasets to benchmark. Only pick pre-stratified ones
datasets = [
    'kinships',
    'nations',
    'umls',
    'countries',
    'codex-small',
    'codex-medium',
    'codex-large',
    'fb15k',
    'fb15k-237',
    'wn18',
    'wn18-rr',
    'yago310',
    'DBpedia50',
]
#: Error rates to check
error_rates = [1.0, 0.6, 0.5, 0.2, 0.1, 0.01, 0.001, 0.0001, 0.00001]


def main(force: bool = False, precision: Optional[int] = None):
    """Benchmark performance of the bloom filterer."""
    df = get_df(force=force, precision=precision)
    plot_df(df)


def plot_df(df: pd.DataFrame):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharex='all', sharey='all')
    sns.lineplot(data=df, x="error_rate", y="testing", hue='dataset', ax=axes[0])
    sns.lineplot(data=df, x="error_rate", y="validation", hue='dataset', ax=axes[1])

    axes[0].set_ylabel('Observed Error Rate')
    axes[0].set_title('Testing')
    axes[1].set_title('Validation')
    for axis in axes.ravel():
        axis.set_xscale('log')
        axis.set_xlabel('Bloom Filter Error Rate')
    fig.tight_layout()
    fig.savefig(PLOT_PATH)


def get_df(force: bool = False, precision: Optional[int] = None):
    if os.path.exists(RESULTS_PATH) and not force:
        return pd.read_csv(RESULTS_PATH, sep='\t')

    if precision is None:
        precision = DEFAULT_PRECISION

    rows = []
    for dataset in datasets:
        dataset = get_dataset(dataset=dataset)
        for error_rate in error_rates:
            filterer = BloomFilterer(triples_factory=dataset.training, error_rate=error_rate)
            print(dataset.get_normalized_name(), error_rate, filterer)
            row = {
                key: round(float(filterer.contains(batch=value.mapped_triples).float().mean()), precision)
                for key, value in dataset.factory_dict.items()
            }
            row.update({
                'error_rate': error_rate,
                'dataset': dataset.get_normalized_name(),
                'size': filterer.bit_array.numel(),
                'training_triples': dataset.training.num_triples,
                'total_triples': sum(tf.num_triples for tf in dataset.factory_dict.values()),
                'natural_size': humanize.naturalsize(filterer.bit_array.numel() / 8),
            })
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_PATH, sep='\t', index=False)
    return df


if __name__ == '__main__':
    main()
