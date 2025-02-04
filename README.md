<h1 align="center" style="fontsize:50em"><b>FLOW</b></h1>

> **Upweighting Easy Samples in Fine-tuning Mitigates Forgeting**\
> Sunny Sanyal*, Hayden Prairie*, Rudrajit Das*, Ali Kavis*, Sujay Sanghavi\
> Paper: [INSERT PAPER] \

## Installing Locally
Our language and vision experiments were run in seperate environments, and thus we have two different installations.

### Vision Installation

### Language Installation

Run the following script to create the environment necissary to run all of the language model experiments.

```bash
conda crate --name flow python=3.10
conda activate flow
pip install -r requirements.txt
```

To install evaluation functionality, please also run the following:

```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

## Vision Experiments

## Language Experiments

We have three stages to our language experiment pipeline. 

1. Evaluate the temperature for a given dataset-model pair
2. Re-weight a dataset with a given temperature
3. Fine-tune a model with a re-weighted dataset

To evaluate the temperature of a model (once `cd` into the language folder), you can simply run the following script:

```bash
bash scripts/launch_get_temperature.slurm
```

or if using slurm then:


```bash
sbatch scripts/launch_get_temperature.slurm
```

To re-weight a dataset, you can run the following script with (`bash`/`sbatch`):

```bash
(bash/sbatch) scripts/launch_weight_dataset.slurm
```

Finally, to train a model you can run the following script:

```bash
(bash/sbatch) scripts/launch_ft_arithmetic.slurm
```

In each of these scripts, you can set the run configuration at the top of the script. In order to evaluate an run use the following script:

```bash
(bash/sbatch) scripts/launch_eval.slurm
```