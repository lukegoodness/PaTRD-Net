# PaTRD-Net

Source code for running PaTRD-Net.

Datasets, checkpoints, predictions, logs, figures, and result files are not included.

## Project structure

```text
src/models/          model implementations
src/data_provider/   data loaders
src/utils/           time-feature utilities
experiments/         training and evaluation scripts
```

## Installation

```bash
git clone <repository-url>
cd PaTRD-Net
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

## Data preparation

Download the public benchmark datasets from the [Autoformer data repository](https://github.com/thuml/Autoformer). Do not commit downloaded data to this repository.

Place the files under the ignored `dataset/` directory:

```text
dataset/
├── ETT-small/
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
├── weather/weather.csv
├── electricity/electricity.csv
└── traffic/traffic.csv
```

## Running the code

Run all commands from the repository root.

Show the available options for a script:

```bash
python experiments/01_main_benchmark.py --help
```

Run PaTRD-Net on the configured benchmarks:

```bash
python experiments/01_main_benchmark.py --include_extended
```

Run the ablation configurations:

```bash
python experiments/02_ablation_study.py
```

Run the same-protocol baseline implementations:

```bash
python experiments/05_baseline_benchmark.py --baseline all --include_extended
```

Run robustness evaluation:

```bash
python experiments/04_robustness.py --mode all
```

Run hyperparameter sensitivity evaluation:

```bash
python experiments/06_hyperparameter_sensitivity.py
```

Aggregate locally generated output files:

```bash
python experiments/08_merge_results.py --report_dir report
```

Parameter counts are written to the `total_params` column of the CSV produced by `01_main_benchmark.py` and `05_baseline_benchmark.py`; no separate profiling script is needed.

FLOPs and wall-clock latency are intentionally not measured here. Standard FLOP counters cannot see the KAN core, because it uses raw `nn.Parameter` with `F.linear` rather than a registered module, and the B-spline evaluation is memory-bandwidth-bound, so neither quantity tracks the parameter count for this architecture.

Generated files are written to ignored output directories. Review staged files before pushing to ensure that no datasets, results, checkpoints, logs, credentials, local paths, or personal information are included.

## License

See [LICENSE](LICENSE).
