We need to install also : 
- IQ-TREE version 2.2.0 or higher:
https://github.com/iqtree/iqtree2/releases/tag/v2.4.0


For now it doesn't work so please look at 

If you want to generate data, there are few other requirements:

6. Install [g++](https://gcc.gnu.org/) (it may be pre-installed on your system)
7. Install [iqtree](http://www.iqtree.org) (at least versions 2.2.0)


# Generating data
Once you have g++ installed, you can generate data using `sh generation_cpp/gen_random.sh`. 

You should change the path to `iqtree` and you may change the `output` path. 




# Some information

If you want to use the already trained embedding network, you can load the model from the checkpoint `./checkpoints/teddy_no_regression.pt`.