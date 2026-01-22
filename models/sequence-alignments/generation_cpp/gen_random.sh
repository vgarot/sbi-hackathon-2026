#!/bin/bash

start=$(date +%s)
id="example" # id of the simulation

json_path="generation_cpp/conf_test.json" # PATH to your config file

output="./data/$id" # PATH to the output folder

# iqtree="<path_to_iqtree2>" # PATH to iqtree2
iqtree="/home/vincent/PhD/code/teddy/deepdynamics/iqtree/iqtree-2.2.2.6-Linux/bin/iqtree2"


mkdir ${output}
mkdir ${output}/trees
mkdir ${output}/seq

cp ${json_path} ${output}/

set -x
g++ -fopenmp generation_cpp/generation_bds_trees.cpp -o $output/generation_bds_trees

# ## 1. Generate trees
${output}/generation_bds_trees ${json_path} $output 

# ## 2. Generate sequences
tail -n +2 $output/design.csv | parallel --colsep ";" --eta --progress --bar \
    "if [ {4} -eq 0 ]; then continue; else $iqtree --alisim $output/seq/{10}_seq -m {8} --length {9} --branch-scale {7} -t $output/trees/{10} -af fasta > /dev/null; fi"


set +x
end=$(date +%s)

echo "Time elapsed for script: $(($end - $start)) s"
