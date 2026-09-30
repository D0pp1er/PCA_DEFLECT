## Running the Project

To run the main script with different attacks, defenses, and datasets:

```bash
python main_runner.py --attack dba --defense pca-deflect --dataset mnist 
```

### Redirect Output to File

To redirect all output (both standard output and errors) of the script to a file, use:

```bash
python3 main_runner.py --attack badnet --defense pca-deflect --dataset mnist > badnet_pca_mnist_output.txt 2>&1
```

This will save all printed output and error messages to `badnet_pca_mnist_output.txt`.

## Reproducing Experiments

For detailed guidance on reproducing experimental results, including:
- Recommended DBSCAN epsilon values for different attacks
- Attack-specific configurations and defense compatibility
- Complete command examples for all supported attacks and defenses
- Defense parameter configurations

**See [Experiment.md](Experiment.md)** for comprehensive experimental setup and reproduction procedures.

### Quick Reference

**Supported Attacks:**
- Data Poisoning: BadNet, DBA, EdgeCase
- Model Poisoning: CerP, ConstrainScale

**Available Defenses:**
- `pca-deflect`: PCA-Deflect defense (recommended for all attacks)
- `nab`: Neural Attention-based Backdoor defense
- `npd`: Neural Polarizer Defense
- `mean`: Baseline federated averaging (no defense)

**Supported Datasets:**
- `mnist`: MNIST handwritten digits
- `fmnist`: Fashion-MNIST clothing items
- `emnist`: Extended MNIST letters and digits
- `cifar`: CIFAR-10 natural images

For attack-specific command examples and optimal parameter configurations, refer to [Experiment.md](Experiment.md).

## Project Structure

```
.
├── README.md
├── Experiment.md
├── config.py
├── main_runner.py
├── attacks
│   └── DBA
│       ├── helper.py
│       ├── image_helper.py
│       ├── image_train.py
│       ├── main.py
│       ├── test.py
│       ├── train.py
│       └── utils
│           ├── cifar_params.yaml
│           ├── csv_record.py
│           └── utils.py
├── defenses
│   └── pca_deflect.py
└── data
    └── MNIST
        └── raw
            ├── t10k-images-idx3-ubyte
            ├── t10k-images-idx3-ubyte.gz
            ├── t10k-labels-idx1-ubyte
            ├── t10k-labels-idx1-ubyte.gz
            ├── train-images-idx3-ubyte
            ├── train-images-idx3-ubyte.gz
            ├── train-labels-idx1-ubyte
            └── train-labels-idx1-ubyte.gz
```

### Directory Descriptions

- **attacks/DBA/**: Contains the DBA (Dummy Bit Attack) implementation
- **defenses/**: Contains defense mechanisms, including PCA-DEFLECT
- **data/**: Contains datasets (MNIST in this case)
- **main_runner.py**: Main entry point for running attacks and defenses
- **config.py**: Configuration settings
- **Experiment.md**: Detailed experimental configuration and reproduction guide
