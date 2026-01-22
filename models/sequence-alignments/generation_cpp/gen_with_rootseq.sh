#!/bin/bash

start=$(date +%s)

id=$1 # id of the simulation

json_path=$2

# output="/home/vincent/PhD/code/teddy2/teddy/data/$id" # output file 
output="/lustre/fsn1/projects/rech/xzo/uhd27ew/$id"

# iqtree="/home/vincent/PhD/code/teddy/deepdynamics/iqtree/iqtree-2.2.2.6-Linux/bin/iqtree2" # path to iqtree
iqtree="/lustre/fswork/projects/rech/xzo/uhd27ew/iqtree/iqtree2"

# rootseq="/linkhome/rech/genggl01/uhd27ew/hcv/short_hcv.fasta,1a.TH.2015.HIVNAT_15.05.MN807647"
# rootseq="/home/vincent/PhD/code/teddy/hcv/init_short_hcv_nt.fasta,1a.US.1977.H77.NC_004102"


mkdir ${output}
mkdir ${output}/trees
mkdir ${output}/seq


cp ${json_path} ${output}/

set -x
g++ -fopenmp generation_cpp/generation_bds_trees.cpp -o $output/generation_bds_trees

# ## 1. Générer les arbres
${output}/generation_bds_trees ${json_path} $output


# # 2. Générer les séquences
# awk -F";" -v output=$output -v iqtree=$iqtree '(NR>1) {
#  system(iqtree " --alisim \"" output "/seq/" $10 "_seq\" -m \"" $8 "\" --seqtype DNA --branch-scale " $7 " -t \"" output "/trees/" $10 "\" -af fasta --root-seq \"" $13 "\" > /dev/null")
#  }' $output/design.csv

## 2. Générer les séquences
tail -n +2 $output/design.csv | parallel --colsep ";" --jobs $SLURM_CPUS_PER_TASK \
    "if [ {4} -eq 0 ]; then continue; else $iqtree --alisim $output/seq/{10}_seq -m {8} --branch-scale {7} -t $output/trees/{10} -af fasta --root-seq {13} > /dev/null; fi"


set +x
end=$(date +%s)

echo "Time elapsed for script: $(($end - $start)) s"
