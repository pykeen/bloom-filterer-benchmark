import humanize
import pandas as pd

from pykeen.datasets import get_dataset
from pykeen.sampling.filtering import BloomFilterer


def main(precision: int = 5):
    """Benchmark performance of the bloom filterer."""
    # Only pick pre-stratified ones
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
    error_rates = [0.5, 0.2, 0.1, 0.01, 0.001, 0.0001, 0.00001]

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
    df.to_csv('bloom_results.tsv', sep='\t', index=False)


if __name__ == '__main__':
    main()
