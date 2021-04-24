import itertools as itt
import os
import time
from typing import Optional

import click
import humanize
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pykeen.datasets import get_dataset
from pykeen.sampling.filtering import BloomFilterer
from tqdm import tqdm

HERE = os.path.abspath(os.path.dirname(__file__))
RESULTS_PATH = os.path.join(HERE, 'results.tsv')
PLOT_SVG_PATH = os.path.join(HERE, 'plot.svg')
PLOT_PNG_PATH = os.path.join(HERE, 'plot.png')

DEFAULT_PRECISION = 5
DEFAULT_TRIALS = 10

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
error_rates = [1.0, 0.8, 0.6, 0.5, 0.2, 0.1, 0.01, 0.001, 0.0001, 0.00001]


@click.command()
@click.option('--force', is_flag=True)
@click.option('--trials', type=int, default=DEFAULT_TRIALS, show_default=True)
@click.option('--precision', type=int, default=DEFAULT_PRECISION, show_default=True)
def main(force: bool, trials: int, precision: int):
    """Benchmark performance of the bloom filterer."""
    df = get_df(force=force, trials=trials, precision=precision)
    plot_df(df)


def plot_df(df: pd.DataFrame):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), sharex='all', sharey='all')
    sns.lineplot(data=df, x="error_rate", y="testing", hue='dataset', ax=axes[0])
    sns.lineplot(data=df, x="error_rate", y="validation", hue='dataset', ax=axes[1])

    axes[0].set_ylabel('Observed Error Rate')
    axes[0].set_title('Testing')
    axes[1].set_title('Validation')
    for axis in axes.ravel():
        axis.set_xscale('log')
        axis.set_xlabel('Bloom Filter Error Rate')
    fig.tight_layout()
    fig.savefig(PLOT_SVG_PATH)
    fig.savefig(PLOT_PNG_PATH, dpi=300)


def get_df(force: bool = False, trials: Optional[int] = None, precision: Optional[int] = None):
    if os.path.exists(RESULTS_PATH) and not force:
        return pd.read_csv(RESULTS_PATH, sep='\t')

    if trials is None:
        trials = DEFAULT_TRIALS
    if precision is None:
        precision = DEFAULT_PRECISION

    rows = []
    outer_it = tqdm(datasets, desc='Datasets')
    for dataset in outer_it:
        dataset = get_dataset(dataset=dataset)
        outer_it.set_postfix({'dataset': dataset.get_normalized_name()})
        inner_it = tqdm(
            itt.product(error_rates, range(trials)),
            desc='Trials',
            total=len(error_rates) * trials,
            leave=False,
        )
        for error_rate, trial in inner_it:
            inner_it.set_postfix({'er': error_rate})
            start_time = time.time()
            filterer = BloomFilterer(triples_factory=dataset.training, error_rate=error_rate)
            end_time = time.time() - start_time
            # print(dataset.get_normalized_name(), error_rate, filterer)
            row = {
                'trial': trial,
                'error_rate': error_rate,
                'dataset': dataset.get_normalized_name(),
                'size': filterer.bit_array.numel(),
                'build_time': end_time,
                'training_triples': dataset.training.num_triples,
                'total_triples': sum(tf.num_triples for tf in dataset.factory_dict.values()),
                'natural_size': humanize.naturalsize(filterer.bit_array.numel() / 8),
            }
            for key, value in dataset.factory_dict.items():
                start_time = time.time()
                res = round(float(filterer.contains(batch=value.mapped_triples).float().mean()), precision)
                end_time = time.time() - start_time
                row[key] = res
                row[f'{key}_time'] = end_time

            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_PATH, sep='\t', index=False)
    return df


if __name__ == '__main__':
    main()
