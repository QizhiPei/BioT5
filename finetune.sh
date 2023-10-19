[ -z "${task}" ] && task=dti
[ -z "${n_node}" ] && n_node=1
[ -z "${n_gpu_per_node}" ] && n_gpu_per_node=1

cd biot5

export TOKENIZERS_PARALLELISM=false

torchrun --nnodes=${n_node} --nproc_per_node=${n_gpu_per_node} main.py \
    task=${task} \
    model.name=google/t5-v1_1-base \
    model.random_init=false \
    model.checkpoint_path=pytorch_model.bin \
    molecule_dict=dict/selfies_dict.txt \
    hydra.run.dir=./logs/dti_local \
    optim.total_steps=50000 optim.warmup_steps=1000 optim.name=adamw \
    optim.lr_scheduler=cosine optim.base_lr=1e-3 \
    seed=42 \
    model.compile=false \
    pred.every_steps=20 logging.every_steps=2 \
    optim.batch_size=48 optim.grad_acc=12 optim.test_bsz_multi=1