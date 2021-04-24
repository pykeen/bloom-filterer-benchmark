import itertools as itt
import os
import time
import timeit
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
CHARTS = os.path.join(HERE, 'charts')
ERROR_PLOT_SVG_PATH = os.path.join(CHARTS, 'errors.svg')
ERROR_PLOT_PNG_PATH = os.path.join(CHARTS, 'errors.png')
SIZE_PLOT_SVG_PATH = os.path.join(CHARTS, 'sizes.svg')
SIZE_PLOT_PNG_PATH = os.path.join(CHARTS, 'sizes.png')
CREATION_TIME_PLOT_SVG_PATH = os.path.join(CHARTS, 'creation_times.svg')
CREATION_TIME_PLOT_PNG_PATH = os.path.join(CHARTS, 'creation_times.png')
LOOKUP_TIME_PLOT_SVG_PATH = os.path.join(CHARTS, 'lookup_times.svg')
LOOKUP_TIME_PLOT_PNG_PATH = os.path.join(CHARTS, 'lookup_times.png')

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
    plot_errors(df)
    plot_size(df)
    plot_creation_time(df)
    plot_lookup_times(df)


def plot_errors(df: pd.DataFrame):
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
    fig.savefig(ERROR_PLOT_SVG_PATH)
    fig.savefig(ERROR_PLOT_PNG_PATH, dpi=300)


def plot_lookup_times(df: pd.DataFrame):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), sharex='all', sharey='all')
    sns.lineplot(data=df, x="error_rate", y="testing_time", hue='dataset', ax=axes[0])
    sns.lineplot(data=df, x="error_rate", y="validation_time", hue='dataset', ax=axes[1])
    axes[0].set_ylabel('Lookup Time (s)')
    axes[0].set_title('Testing')
    axes[1].set_title('Validation')
    for axis in axes.ravel():
        axis.set_xscale('log')
        axis.set_xlabel('Bloom Filter Error Rate')
    fig.tight_layout()
    fig.savefig(LOOKUP_TIME_PLOT_SVG_PATH)
    fig.savefig(LOOKUP_TIME_PLOT_PNG_PATH, dpi=300)


def plot_size(df: pd.DataFrame):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), sharey='all')
    sns.scatterplot(data=df, x="training_triples", y="size", hue='dataset', ax=axes[0])
    sns.lineplot(data=df, x="error_rate", y="size", hue='dataset', ax=axes[1])
    axes[0].set_ylabel('Size (bytes)')
    for axis in axes.ravel():
        axis.set_xscale('log')
    fig.tight_layout()
    fig.savefig(SIZE_PLOT_SVG_PATH)
    fig.savefig(SIZE_PLOT_PNG_PATH, dpi=300)


def plot_creation_time(df: pd.DataFrame):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), sharey='all')
    sns.scatterplot(data=df, x="training_triples", y="time", hue='dataset', ax=axes[0])
    sns.lineplot(data=df, x="error_rate", y="time", hue='dataset', ax=axes[1])
    axes[0].set_ylabel('Creation Time (s)')
    for axis in axes.ravel():
        axis.set_xscale('log')
    fig.tight_layout()
    fig.savefig(CREATION_TIME_PLOT_SVG_PATH)
    fig.savefig(CREATION_TIME_PLOT_PNG_PATH, dpi=300)


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
            # measure creation (=indexing) time
            timer = timeit.Timer(
                stmt="BloomFilterer(triples_factory=factory, error_rate=error_rate)",
                globals=dict(
                    BloomFilterer=BloomFilterer,
                    factory=dataset.training,
                    error_rate=error_rate,
                )
            )
            repetitions, total_time = timer.autorange()
            end_time = total_time / repetitions
            filterer = BloomFilterer(triples_factory=dataset.training, error_rate=error_rate)
            row = {
                'dataset': dataset.get_normalized_name(),
                'training_triples': dataset.training.num_triples,
                'testing_triples': dataset.testing.num_triples,
                'validation_triples': dataset.validation.num_triples,
                'total_triples': sum(tf.num_triples for tf in dataset.factory_dict.values()),
                'trial': trial,
                'error_rate': error_rate,
                'time': end_time,
                'size': filterer.bit_array.numel(),
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
