<h1 align="center" style="fontsize:50em"><b>FLOW</b></h1>

> **Upweighting Easy Samples in Fine-tuning Mitigates Forgeting**\
> Sunny Sanyal*, Hayden Prairie*, Rudrajit Das*, Ali Kavis*, Sujay Sanghavi\
> Paper: [INSERT PAPER] \

## Installing Locally
Our language and vision experiments were run in seperate environments, and thus we have two different installations.

### Vision Installation

```bash
cd vision
conda crate --name flow python=3.9.12
conda activate flow_vision
pip install -r requirements.txt
```

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

The datasets for vision experiments are downloaded using torchvision.datasets, except [stanford cars](https://github.com/cyizhuo/Stanford_Cars_dataset).

FLOW fine-tuning for vision models are performed as follows.

1. Download a pre-trained model to be fine-tuned.
2. Perform linear probing on model from step 1, using a target dataset to develop a linear probe (lp) model.
3. Using the lp model, evaluate the temperature (median lp loss) for a given dataset-model pair. 
4. Re-weight every sample of a target dataset using lp loss and temperature.
5. Finetune the model using sample-wise weighted loss.

One can finetune ResNet-18/ResNet-50 on 6 image classification datasets based on the following steps.

To run full finetuning, you can run the following script:

```bash
bash run_standardfinetune.sh
```

To linear probe a ImageNet-1K pre-trained model, you can run the following script:

```bash
bash run_linearprobing.sh
```

Next we can evaluate the temperature of the dataset model pair using the following script:

```bash
python compute_temp.py --dataset cifar10 --model resnet18 --checkpoint-dir ./checkpoint/linear/resnet18 --loss-save-dir ./logs/ours/train_loss
```

Finetune the full model with sample-wise weighted loss using the following script:

```bash
bash run_flow_round1.sh
```

Next we finetune only the task specific head with regular loss. This is done using the following script:

```bash
bash run_flow_round2.sh
```

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