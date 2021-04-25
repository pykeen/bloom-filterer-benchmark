import pathlib
import timeit
from typing import Any, Iterable, Mapping, Optional, Tuple

import click
import humanize
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from class_resolver import Hint
from docdata import get_docdata
from pykeen.datasets import Dataset, datasets as datasets_dict, get_dataset
from pykeen.sampling.filtering import BloomFilterer, Filterer, filterer_resolver
from torch.utils.benchmark import Timer as TorchTimer
from tqdm import tqdm

HERE = pathlib.Path(__file__).parent
RESULTS_PATH = HERE.joinpath('results.tsv')
COMPARISON_PATH = HERE.joinpath('comparison.tsv')
CHARTS = HERE.joinpath('charts')
CHARTS.mkdir(exist_ok=True)
COMPARISON = CHARTS / 'comparison'
COMPARISON.mkdir(exist_ok=True)
ERROR_PLOT_SVG_PATH = CHARTS.joinpath('errors.svg')
ERROR_PLOT_PNG_PATH = CHARTS.joinpath('errors.png')
SIZE_PLOT_SVG_PATH = CHARTS.joinpath('sizes.svg')
SIZE_PLOT_PNG_PATH = CHARTS.joinpath('sizes.png')
CREATION_TIME_PLOT_SVG_PATH = CHARTS.joinpath('creation_times.svg')
CREATION_TIME_PLOT_PNG_PATH = CHARTS.joinpath('creation_times.png')
LOOKUP_TIME_PLOT_SVG_PATH = CHARTS.joinpath('lookup_times.svg')
LOOKUP_TIME_PLOT_PNG_PATH = CHARTS.joinpath('lookup_times.png')

DEFAULT_PRECISION = 5

sns.set_style('whitegrid')

#: Datasets to benchmark. Only pick pre-stratified ones
datasets = [
    'kinships',
    'nations',
    'umls',
    'countries',
    'codexsmall',
    'codexmedium',
    'codexlarge',
    'fb15k',
    'fb15k237',
    'wn18',
    'wn18rr',
    'yago310',
    'dbpedia50',
]
# Order by increasing number of triples
datasets = sorted(datasets, key=lambda s: get_docdata(datasets_dict[s])['statistics']['triples'])

#: Error rates to check
error_rates = [1.0, 0.8, 0.6, 0.5, 0.2, 0.1, 0.01, 0.001, 0.0001, 0.00001]

HUE_ORDER = ['pythonset', 'bloom']

@click.command()
@click.option('--force', is_flag=True)
@click.option('--test', is_flag=True)
@click.option('--precision', type=int, default=DEFAULT_PRECISION, show_default=True)
def main(force: bool, test: bool, precision: int):
    """Benchmark performance of the bloom filterer."""
    comparison_df = compare_filterers(test=test, force=force)
    comparison_df.sort_values('filterer', ascending=True, inplace=True)
    plot_comparison_setup(comparison_df)
    plot_comparison_lookup_time(comparison_df)
    plot_comparison_errors(comparison_df)
    plot_comparison_2d(comparison_df)

    bloom_benchmark_df = get_bloom_benchmark_df(force=force, precision=precision)
    plot_errors(bloom_benchmark_df)
    plot_size(bloom_benchmark_df)
    plot_creation_time(bloom_benchmark_df)
    plot_lookup_times(bloom_benchmark_df)


def plot_errors(df: pd.DataFrame):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), sharex='all', sharey='all')
    sns.lineplot(data=df, x="error_rate", y="testing", hue='dataset', ax=axes[0])
    sns.lineplot(data=df, x="error_rate", y="validation", hue='dataset', ax=axes[1])
    axes[0].set_ylabel('Observed Error Rate')
    axes[0].set_title('Testing')
    axes[1].set_title('Validation')
    for axis in axes.ravel():
        axis.set_xscale('log')
        # When switching to log scale, it's easier to see the linear relationship,
        # but the missing values and 0 values become a problem
        # axis.set_yscale('log')
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
        axis.set_yscale('log')
        axis.set_xlabel('Bloom Filter Error Rate')
    fig.tight_layout()
    fig.savefig(LOOKUP_TIME_PLOT_SVG_PATH)
    fig.savefig(LOOKUP_TIME_PLOT_PNG_PATH, dpi=300)


def plot_size(df: pd.DataFrame):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), sharey='all')
    sns.scatterplot(data=df, x="training_triples", y="size", hue='error_rate', alpha=0.8, ax=axes[0])
    sns.lineplot(data=df, x="error_rate", y="size", hue='dataset', ax=axes[1])
    axes[0].set_ylabel('Size (bytes)')
    for axis in axes.ravel():
        axis.set_xscale('log')
        axis.set_yscale('log')
    fig.tight_layout()
    fig.savefig(SIZE_PLOT_SVG_PATH)
    fig.savefig(SIZE_PLOT_PNG_PATH, dpi=300)


def plot_creation_time(df: pd.DataFrame):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), sharey='all')
    sns.scatterplot(data=df, x="training_triples", y="time", hue='error_rate', alpha=0.8, ax=axes[0])
    sns.lineplot(data=df, x="error_rate", y="time", hue='dataset', ax=axes[1])
    axes[0].set_ylabel('Creation Time (s)')
    for axis in axes.ravel():
        axis.set_xscale('log')
        axis.set_yscale('log')
    fig.tight_layout()
    fig.savefig(CREATION_TIME_PLOT_SVG_PATH)
    fig.savefig(CREATION_TIME_PLOT_PNG_PATH, dpi=300)


def plot_comparison_setup(df: pd.DataFrame):
    indexing_df = df.loc[df['operation'] == 'index', ['dataset', 'filterer', 'time', 'num_triples']]

    fig, axes = plt.subplots(figsize=(10, 4))
    sns.scatterplot(
        data=indexing_df,
        y='time',
        x='num_triples',
        hue='filterer',
        hue_order=HUE_ORDER,
        style='filterer',
        style_order=HUE_ORDER,
        ax=axes,
        alpha=0.8,
    )
    axes.set_ylabel('Index Time (s)')
    axes.set_xlabel('Number Triples')
    axes.set_xscale('log')
    axes.set_yscale('log')

    fig.savefig(COMPARISON / 'setup.svg')
    fig.savefig(COMPARISON / 'setup.png', dpi=300)


def plot_comparison_lookup_time(df: pd.DataFrame):
    columns = ['dataset', 'filterer', 'time', 'num_triples']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey='all')
    for key, ax in zip(('testing', 'validation'), axes):
        data = df.loc[df['subset'] == key, columns]
        sns.scatterplot(
            data=data,
            x='num_triples',
            hue='filterer',
            hue_order=HUE_ORDER,
            style='filterer',
            style_order=HUE_ORDER,
            y='time',
            alpha=0.8,
            ax=ax,
        )
        ax.set_title(key.capitalize())
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel('Number Triples')

    axes[0].set_ylabel('Lookup Time (s)')
    axes[1].set_ylabel('')

    fig.tight_layout()
    fig.savefig(COMPARISON / 'lookup_times.svg')
    fig.savefig(COMPARISON / 'lookup_times.png', dpi=300)


def plot_comparison_errors(df: pd.DataFrame):
    columns = ['dataset', 'filterer', 'observed_error_rate', 'num_triples']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex='all')
    for key, ax in zip(('testing', 'validation'), axes):
        data = df.loc[df['subset'] == key, columns]
        data['adj_observed_error_rate'] = data['observed_error_rate'] + 1 / data['num_triples']
        sns.scatterplot(
            data=data,
            x='num_triples',
            y='adj_observed_error_rate',
            hue='filterer',
            hue_order=HUE_ORDER,
            style='filterer',
            style_order=HUE_ORDER,
            alpha=0.8,
            ax=ax,
        )
        ax.set_title(key.capitalize())
        ax.set_xlabel('Number Triples')
        ax.set_yscale('log')
        ax.set_xscale('log')

    axes[0].set_ylabel('Adjusted Observed Error Rate')
    axes[1].set_ylabel('')

    fig.tight_layout()
    fig.savefig(COMPARISON / 'errors.svg')
    fig.savefig(COMPARISON / 'errors.png', dpi=300)


def plot_comparison_2d(df: pd.DataFrame):
    columns = ['dataset', 'filterer', 'observed_error_rate', 'time', 'num_triples']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey='all')
    for key, ax in zip(('testing', 'validation'), axes):
        data = df.loc[df['subset'] == 'testing', columns]
        data['adj_observed_error_rate'] = data['observed_error_rate'] + 1 / data['num_triples']
        sns.scatterplot(
            data=data,
            x='adj_observed_error_rate',
            y='time',
            hue='filterer',
            hue_order=HUE_ORDER,
            style='filterer',
            style_order=HUE_ORDER,
            alpha=0.8,
            ax=ax,
        )
        ax.set_title(key.capitalize())
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel('Adjusted Observed Error Rate')

    axes[0].set_ylabel('Lookup Time (s)')

    fig.tight_layout()
    fig.savefig(COMPARISON / 'errors_2d.svg')
    fig.savefig(COMPARISON / 'errors_2d.png', dpi=300)


def compare_filterers(test: bool = False, force: bool = False):
    if COMPARISON_PATH.exists() and not force:
        return pd.read_csv(COMPARISON_PATH, sep='\t')

    rows = [
        row
        for dataset in iter_datasets(test=test)
        for filterer, filterer_kwargs in iter_experiments()
        for row in benchmark_filterer(dataset=dataset, filterer=filterer, filterer_kwargs=filterer_kwargs)
    ]
    df = pd.DataFrame(rows)
    df.to_csv(COMPARISON_PATH, sep='\t', index=False)
    return df


def iter_experiments() -> Iterable[Tuple[str, Mapping[str, Any]]]:
    experiments = [
        # ('default', {}),
        ('pythonset', {}),
        *(
            ('bloom', dict(error_rate=error_rate))
            for error_rate in error_rates
        )
    ]
    it = tqdm(experiments, desc='Experiments', leave=False)
    for filterer, filter_kwargs in it:
        it.set_postfix(filterer=filterer, **filter_kwargs)
        yield filterer, filter_kwargs


def benchmark_filterer(
    dataset: Dataset,
    filterer: Hint[Filterer],
    filterer_kwargs: Optional[Mapping[str, Any]] = None,
) -> Iterable[Mapping[str, Any]]:
    """Benchmark a filterer."""
    filterer_kwargs = filterer_kwargs or {}

    # include some metadata into each entry
    kwargs = dict(
        dataset=dataset.get_normalized_name(),
        filterer=filterer,
        **filterer_kwargs,
    )

    filterer_cls = filterer_resolver.lookup(filterer)
    tqdm.write(f'[{filterer_cls.__name__}] measure creation (=indexing) time')
    timer = TorchTimer(
        stmt="filterer_cls(triples_factory=factory, **kwargs)",
        globals=dict(
            filterer_cls=filterer_cls,
            factory=dataset.training,
            kwargs=filterer_kwargs,
        )
    )
    measurement = timer.blocked_autorange()
    yield dict(
        operation="index",
        subset="train",
        time=measurement.median,
        num_triples=dataset.training.num_triples,
        **kwargs,
    )

    # instantiate filterer for further tests
    filterer = filterer_resolver.make(filterer, pos_kwargs=filterer_kwargs, triples_factory=dataset.training)
    for key, value in dataset.factory_dict.items():
        if key == 'training':
            continue
        tqdm.write(f'[{filterer}] measure inference time ({key})')
        timer = TorchTimer(
            stmt="filterer(mapped_triples)",
            globals=dict(
                filterer=filterer,
                mapped_triples=value.mapped_triples,
            )
        )
        measurement = timer.blocked_autorange()

        # check for correctness
        error_rate = float((~filterer(value.mapped_triples)[1]).float().mean().item())
        yield dict(
            operation="inference",
            subset=key,
            time=measurement.median,
            num_triples=value.num_triples,
            observed_error_rate=error_rate,
            **kwargs,
        )


def get_bloom_benchmark_df(force: bool = False, precision: Optional[int] = None):
    if RESULTS_PATH.is_file() and not force:
        return pd.read_csv(RESULTS_PATH, sep='\t')

    if precision is None:
        precision = DEFAULT_PRECISION

    rows = []
    for dataset in iter_datasets():
        inner_it = tqdm(
            error_rates,
            desc='Error Rates',
            leave=False,
        )
        for error_rate in inner_it:
            inner_it.set_postfix({'er': error_rate})
            tqdm.write('measure creation (=indexing) time')
            timer = timeit.Timer(
                stmt="filterer_cls(triples_factory=triples_factory, error_rate=error_rate)",
                globals=dict(
                    filterer_cls=BloomFilterer,
                    triples_factory=dataset.training,
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
                'error_rate': error_rate,
                'time': end_time,
                'size': filterer.bit_array.numel(),
                'natural_size': humanize.naturalsize(filterer.bit_array.numel() / 8),
            }
            for key, value in dataset.factory_dict.items():
                tqdm.write(f'measure inference time ({key})')
                timer = timeit.Timer(
                    stmt="filterer.contains(batch=mapped_triples)",
                    globals=dict(
                        filterer=filterer,
                        mapped_triples=value.mapped_triples,
                    )
                )
                repetitions, total_time = timer.autorange()
                end_time = total_time / repetitions
                # check for correctness
                res = round(float(filterer.contains(batch=value.mapped_triples).float().mean()), precision)
                row[key] = res
                row[f'{key}_time'] = end_time

            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_PATH, sep='\t', index=False)
    return df


def iter_datasets(test: bool = False) -> Iterable[Dataset]:
    it = tqdm(datasets[:5] if test else datasets, desc='Datasets')
    for dataset in it:
        dataset_instance = get_dataset(dataset=dataset)
        it.write(f'loaded {dataset_instance.get_normalized_name()}')
        it.set_postfix(dataset=dataset_instance.get_normalized_name())
        yield dataset_instance


if __name__ == '__main__':
    main()
