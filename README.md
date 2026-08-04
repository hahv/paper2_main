# iNet Evaluation Scripts on UFireIndoorVideo datasets

This repository contains the evaluation scripts (python) for the iNet model
fire/smoke detection on UFireIndoorVideo datasets.

Pls follow the following steps to run the evaluation scripts:
> [!Note] This repository using `cuda 12.9` so make sure you have the correct version of `cuda` installed in your system. 
> Also it is recommended to run it in `WSL2` environment with `Ubuntu 24.04` OS.
> You can install `Ubuntu 24.04` in `WSL2` by following this [instruction
> link](https://ubuntu.com/wsl/docs/stable/howto/install-ubuntu-wsl2/). Although
> we still can run this in Windows environment.


```bash
   nvcc --version
```

## 1. Install python dependencies

We use `uv` to install the dependencies. First, install `uv` (depending on your
dev environment/OS) via this [instruction
link](https://docs.astral.sh/uv/getting-started/installation/)
Then we can create the venv folder `.venv` by running the following command:

```bash
   uv sync --extra gpu # must has --extra gpu to install pytorch with cuda support
```

## 2. Change the values in config file

The `config/config.yaml` file contains the configuration parameters for the
evaluation scripts. You need to change the values in this file according to your
requirements. The parameters are explained in the comments in the config file.

## 3. Run the evaluation scripts

Run the following command to run the evaluation scripts:

```bash
   uv run python run.py --c config/config.yaml
```

or activate the venv and run the script directly:

```bash
   source .venv/bin/activate # or .venv/Scripts/activate for Windows
   python run.py --c config/config.yaml
```

> [!Note] Each config `yaml` file will generate a hash value based on the config
> values, including: [general, dataset, modelCfg, methodCfg, evalCfg]. The hash value will be used to