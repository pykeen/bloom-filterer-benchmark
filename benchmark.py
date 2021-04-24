import pathlib
import timeit
from typing import Any, Iterable, Mapping, Optional

import click
import humanize
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from class_resolver import Hint
from pykeen.datasets import Dataset, get_dataset
from pykeen.sampling.filtering import BloomFilterer, Filterer, filterer_resolver
from torch.utils.benchmark import Timer as TorchTimer
from tqdm import tqdm

HERE = pathlib.Path(__file__).parent
RESULTS_PATH = HERE.joinpath('results.tsv')
CHARTS = HERE.joinpath('charts')
ERROR_PLOT_SVG_PATH = CHARTS.joinpath('errors.svg')
ERROR_PLOT_PNG_PATH = CHARTS.joinpath('errors.png')
SIZE_PLOT_SVG_PATH = CHARTS.joinpath('sizes.svg')
SIZE_PLOT_PNG_PATH = CHARTS.joinpath('sizes.png')
CREATION_TIME_PLOT_SVG_PATH = CHARTS.joinpath('creation_times.svg')
CREATION_TIME_PLOT_PNG_PATH = CHARTS.joinpath('creation_times.png')
LOOKUP_TIME_PLOT_SVG_PATH = CHARTS.joinpath('lookup_times.svg')
LOOKUP_TIME_PLOT_PNG_PATH = CHARTS.joinpath('lookup_times.png')

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
error_rates = [1.0, 0.8, 0.6, 0.5, 0.2, 0.1, 0.01, 0.001, 0.0001, 0.00001]


@click.command()
@click.option('--force', is_flag=True)
@click.option('--precision', type=int, default=DEFAULT_PRECISION, show_default=True)
def main(force: bool, precision: int):
    """Benchmark performance of the bloom filterer."""
    df = get_df(force=force, precision=precision)
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
    sns.scatterplot(data=df, x="training_triples", y="size", hue='error_rate', ax=axes[0])
    sns.lineplot(data=df, x="error_rate", y="size", hue='dataset', ax=axes[1])
    axes[0].set_ylabel('Size (bytes)')
    for axis in axes.ravel():
        axis.set_xscale('log')
    fig.tight_layout()
    fig.savefig(SIZE_PLOT_SVG_PATH)
    fig.savefig(SIZE_PLOT_PNG_PATH, dpi=300)


def plot_creation_time(df: pd.DataFrame):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), sharey='all')
    sns.scatterplot(data=df, x="training_triples", y="time", hue='error_rate', ax=axes[0])
    sns.lineplot(data=df, x="error_rate", y="time", hue='dataset', ax=axes[1])
    axes[0].set_ylabel('Creation Time (s)')
    for axis in axes.ravel():
        axis.set_xscale('log')
    fig.tight_layout()
    fig.savefig(CREATION_TIME_PLOT_SVG_PATH)
    fig.savefig(CREATION_TIME_PLOT_PNG_PATH, dpi=300)


def benchmark_filterer(
    dataset: Dataset,
    filterer: Hint[Filterer],
    filterer_kwargs: Optional[Mapping[str, Any]] = None,
) -> Iterable[Mapping[str, Any]]:
    """Benchmark a filterer."""
    filterer_kwargs = filterer_kwargs or {}

    # include some meta-data into each entry
    kwargs = dict(
        dataset=dataset.get_normalized_name(),
        filterer=filterer,
        **filterer_kwargs,
    )

    # measure creation (=indexing) time
    filterer_cls = filterer_resolver.lookup(filterer)
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
        # measure inference time
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


def get_df(force: bool = False, precision: Optional[int] = None):
    if RESULTS_PATH.is_file() and not force:
        return pd.read_csv(RESULTS_PATH, sep='\t')

    if precision is None:
        precision = DEFAULT_PRECISION

    rows = []
    outer_it = tqdm(datasets, desc='Datasets')
    for dataset in outer_it:
        dataset = get_dataset(dataset=dataset)
        outer_it.set_postfix({'dataset': dataset.get_normalized_name()})
        inner_it = tqdm(
            error_rates,
            desc='Error Rates',
            leave=False,
        )
        for error_rate in inner_it:
            inner_it.set_postfix({'er': error_rate})
            # measure creation (=indexing) time
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
                # measure inference time
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


if __name__ == '__main__':
    main()
