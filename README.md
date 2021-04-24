# bloom-filterer-benchmark

The bloom filterer was introduced in [PyKEEN #401](https://github.com/pykeen/pykeen/pull/401)
by Max Berrendorf ([@mberr](https://github.com/mberr)). This benchmarking can be rerun
with `python benchmark.py --force`.

Benchmarking over several datasets of varying size shows suggests that there isn't a large size-dependence on the
relationship between the bloom filter's
`error_rate` parameter and the actual error observed on either the testing or validation sets.

<img src="charts/errors.svg" />

As expected, the time for checking the triples decreases with an increased nominal error rate.

<img src="charts/lookup_times.svg" />

Datasets with a larger number of triples take longer to create. The time to create a bloom filter also decreases as the
nominal error rate increases, except there's some funny activity as it approaches error rate = 1.0.

<img src="charts/creation_times.svg" />

The size of the bloom filter increases with larger number of training triples, but also varies exponentially with the
error rate.

<img src="charts/sizes.svg" />
