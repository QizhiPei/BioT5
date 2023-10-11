ft_task=$1

cd BioT5

python main.py \
    task=${ft_task} \
    model.name=google/t5-v1_1-base \
    model.random_init=false \
    model.checkpoint_path=/BioT5/checkpoints/unimpt/gpu8_bsz96_acc3_ts200000_lr1e-2_seed42_append/checkpoint-pt-115000/pytorch_model.bin \
    molecule_dict=dict/selfies_dict.txt \
    hydra.run.dir=./logs/dti_local \
    optim.total_steps=50000 optim.warmup_steps=1000 optim.name=adamw \
    optim.lr_scheduler=cosine optim.base_lr=1e-3 \
    seed=42 \
    model.compile=false \
    pred.every_steps=20 logging.every_steps=2 \
    optim.batch_size=48 optim.grad_acc=12 optim.test_bsz_multi=1