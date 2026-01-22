# Teddy

Teddy is a deep-learning tool that estimates epidemiological parameters from pathogen sequence alignment and sampling dates.

It estimates two of the three parameters of the birth-death-sampling model ([Stadler](https://www.sciencedirect.com/science/article/abs/pii/S0022519310004765)) without contemporary sampling. Namely, it produces estimates of the reproductive number $R_0$ and the mean duration of infection $\delta$ according that sampling proportion is given in input. Moreover, credible intervals are also estimated.

For more details, check our amazing paper ......

# Installation

1. Clone this repository
2. Create a conda environment by running `conda create -p ./teddy_env -c defaults python=3.12`
3. Activate the environment by running `conda activate teddy_env`
4. Install the required packages by running `pip install --no-cache -r requirements.txt`
5. In case you want to reproduce results and use the provided `.ipynb` files, run `pip install --no-cache -r requirements_plots.txt`


If you want to generate data, there are few other requirements:

6. Install [g++](https://gcc.gnu.org/) (it may be pre-installed on your system)
7. Install [iqtree](http://www.iqtree.org) (at least versions 2.2.0)



# Loading Teddy

To load Teddy, you can use the following code:
```python
from teddy.lightning.teddy_lightning import TeddyLightning
ckpt_path = "./checkpoints/teddy_0.ckpt" # path to the checkpoint
teddy = TeddyLightning.load_from_checkpoint(ckpt_path)
``` 
# Data for Teddy
The input data for Teddy is reduced dated Multiple Sequence Alignment and a sampling proportion.
To obtain this data, you can use the `prep_data` function in the `teddy/data/prep_data.py` file. This function takes as input a path to a fasta file (with dates of sampling in the name of each sequence as `name|date`), the alphabet you want to decypher with (from the `teddy/data/Alphabet.Alphabet` class) and the sampling proportion. 

```python
from teddy.data.prep_data import prep_data
from teddy.data.Alphabet import Alphabet
import torch
fasta_path = "./data/example/seq/BD__0.146006__0.116316__0.243933.newick_seq.fa"
alphabet = Alphabet(list("ATGCX-"))
sampling_proportion = 0.243933
data,shape = prep_data(fasta_path,
                        alphabet,
                        sampling_proportion
                        )
data = data.unsqueeze(0) # Add a batch dimension
shape = torch.Tensor(shape).unsqueeze(0) # Add a batch dimension
teddy.eval()
output = teddy.forward(data,shape)
```


# Generating data
Once you have g++ installed, you can generate data using `sh generation_cpp/gen_random.sh`. 

You should change the path to `iqtree` and you may change the `output` path. 

# Running predictions

## On a simulated dataset
The easiest way to run predictions on the simulated dataset is to use the `scripts/predict_dir.py` script.
You can run it as follows:
```bash
python scripts/predict_dir.py --data_dir ./data/test/seq/ --model_path ./checkpoints/teddy_0.ckpt -p ./test.csv --limit_size 250 --batch_size 1
```
This will load the model from the checkpoint and run predictions on all the fasta files in the `data_dir` folder. The results will be saved in the `test.csv` file.

You can also specify other parameters as `--limit_size` to limit the size of the sequences after deleting constant sites over the MSA, `--batch_size` to set the batch size for predictions, `--scale_dates` to scale the dates,`--resample n` and `--size_samples s1 s2` to resample n times sequences of size between s1 and s2 from each sequence. The argument `--accelerator` allows to choose the accelerator to use (by default "cpu" is used, but you can also use "auto" if you have a GPU available).

## On a test data
You can also run predictions on a single fasta file using the `scripts/predict_for_true.py` script.
You can run it as follows:
```bash
python scripts/predict_for_true.py --fasta_path <path_to_fasta> --dates_path <path_to_dates_file> --sampling_prop <sampling_proportion> --model_path ./checkpoints/teddy_hcv.ckpt -p ./test_true.csv --limit_size 250
```
This will load the model from the checkpoint and run predictions on the fasta file. The results will be saved in the `test_true.csv` file.

# Training from scratch
The scripts are written to run on GPUs. You can train Teddy from scratch using the `scripts/train_teddy.py` script. 
You can run it as follows:
```bash
python scripts/train_teddy.py --config_path ./configs_training/teddy_0.json
```
This will train Teddy using the configuration file `teddy_0.json`. You should modify the configuration file to set the paths to your data and where you want to save the logs and checkpoints.
Be aware that training Teddy requires a lot of computational resources and time. The given parameters are for training on a H100 GPU with 80GB of VRAM. You may need to adjust the batch size and other parameters depending on your hardware.

# Fine-tuning on a dataset
You can fine-tune a pre-trained Teddy model on a specific dataset using the `scripts/finetune_teddy.py` script.
You can run it as follows:
```bash
python scripts/finetune_teddy.py --config_path ./configs_training/finetune_hcv.json
```
This will fine-tune Teddy using the configuration file `finetune_hcv.json`. You should modify the configuration file to set the paths to your data, the pre-trained model checkpoint and where you want to save the logs and checkpoints.
