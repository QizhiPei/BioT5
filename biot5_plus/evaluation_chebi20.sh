ckpt_path=path_to_your_model
data_dir=../data/splits_plus/mol_text
task_dir=../data/tasks_plus
run_dir=../logs/chebi20_test

echo mol2text
python main.py \
        predict_only=true \
        task=ft_mol_multi \
        test_task=mol2text \
        model.name=google/t5-v1_1-base \
        model.checkpoint_path=${ckpt_path} \
        optim.batch_size=64 optim.grad_acc=1 \
        data.task_dir=${task_dir} \
        data.data_dir=${data_dir}/mol2text \
        hydra.run.dir=${run_dir}/mol2text

echo text2mol
python main.py \
        predict_only=true \
        task=ft_mol_multi \
        test_task=text2mol \
        model.name=google/t5-v1_1-base \
        model.checkpoint_path=${ckpt_path} \
        optim.batch_size=64 optim.grad_acc=1 \
        data.task_dir=${task_dir} \
        data.data_dir=${data_dir}/text2mol \
        data.max_seq_len=1024 data.max_target_len=384 \
        hydra.run.dir=${run_dir}/text2mol
