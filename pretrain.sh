set -x
export TOKENIZERS_PARALLELISM=false
cd BioT5

n_gpu=$(nvidia-smi -L | wc -l)
bsz=96
grad_acc=3
seed=128
ts=350000
lr=1e-2

pre=/BioT5
save_dir=${pre}/checkpoints/unimpt/gpu${n_gpu}_bsz${bsz}_acc${grad_acc}_ts${ts}_lr${lr}_seed${seed}

head ${pre}/data/unimpt/mol_text_nolap.tsv -n 1
head ${pre}/data/unimpt/pro_text.tsv -n 1

torchrun --nnodes=1 --nproc_per_node=${n_gpu} main.py \
    optim.batch_size=${bsz} \
    optim.total_steps=${ts} \
    optim.grad_acc=${grad_acc} \
    optim.base_lr=${lr} \
    seed=${seed} \
    hydra.run.dir=./logs/itp_local \
    pair_data_dir=${pre}/data/unimpt \
    incontext_data_dir=${pre}/data/pubmed/annotation_v1.1_replaced_selfies \
    model.name=google/t5-v1_1-base \
    model.compile=True
