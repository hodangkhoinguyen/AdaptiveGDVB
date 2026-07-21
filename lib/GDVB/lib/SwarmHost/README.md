# SwarmHost
Spawns Locusts.



## 1 INSTALLATION

### I. Install environment
Install the conda environment
`conda env create --name SwarmHost -f env.yml`

### II. Install verifiers
Clone your preferred verifier(`VERIFIER`) into the `lib` directory.
`mkdir lib`
`cd lib`
`git clone [VERIFIER_REPO_LINK]`
e.g.,
`git clone https://github.com/Verified-Intelligence/alpha-beta-CROWN`

Install the conda environment of your preferred verifier(`VERIFIER`)
`conda env create --name [VERIFIER] -f envs/[VERIFIER].yml`
e.g.,
`conda env create --name abcrown -f envs/abcrown.yml` 
    


#### a) Supported verifiers
A list of supported [VERIFIER]s and their [VERIFIER_REPO_LINK]s are shown as follows:
    
> [$\alpha$-$\beta$-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN)
> [mnbab]
> [nnenum]
> [verinet]
> [neuralsat]
> [veristable]
> [PyRAT](https://git.frama-c.com/pub/pyrat) (see note below -- proprietary CEA license)

##### PyRAT-specific setup
PyRAT's source is proprietary (CEA license) but the repository is publicly
cloneable; review the license at [pyrat-analyzer.com](https://pyrat-analyzer.com/)
before depending on it.

```shell
mkdir -p lib
git clone https://git.frama-c.com/pub/pyrat.git lib/pyrat
cd lib/pyrat
conda env create --name pyrat -f ../../envs/pyrat.yml
conda run -n pyrat pip install -e .
```

Verified working with `python=3.10.17`; other minor versions may hit a "bad
magic number" error since PyRAT ships precompiled `.pyc` bytecode tied to a
specific CPython build. If `numpy>=2.0` gets pulled in by another dependency,
downgrade it (`pip install "numpy<2.0"`) -- PyRAT's own `pyproject.toml`
requires `numpy<2.0`.

Smoke-test the install:
```shell
conda run -n pyrat pyrat --model_path <abs path to .onnx> --property_path <abs path to .vnnlib> --timeout 30 --domains poly
```
It prints a line like `Result = True, Time = 0.04 s, Safe space = 0.00 %,
number of analysis = 1` -- this is what `swarm_host/verifiers/pyrat` parses.
`Result` is one of PyRAT's own status values (`True`/`False`/`Unknown`/
`Error`/`Timeout`), mapped to `unsat`/`sat`/`unknown`/`error`/`timeout`
respectively (`True` = property holds/no counterexample = unsat; `False` =
counterexample found = sat).

Use **absolute paths** for `--model_path`/`--property_path`/`--log_dir` --
PyRAT resolves relative paths against its own install directory rather than
the caller's working directory, which is easy to get bitten by.

### III. Get miscellaneous tools
Download the resource monitor from the DNNV framework.
`wget https://github.com/dlshriver/dnnv/blob/main/tools/resmonitor.py lib\`


### 2 USAGE

#### I. Prepare your neural network model and property

##### a) neural network model format
This tool supports the standard ONNX model format defined in [VNN-LIB](https://www.vnnlib.org). Prepare your neural network model in this format.

##### b) property
This tool also uses the VNNLIB standard(.vnnlib) for the property language. You can either prepare your own properties by following the standard or use the built-in LocalRobustness properties generator.

##### c) a list of other options
> [--time]
> [--memory]