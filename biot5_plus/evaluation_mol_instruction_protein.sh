ckpt_path=path_to_your_model
data_dir=../data/splits_plus/molinst/eval
task_dir=../data/tasks_plus
run_dir=../logs/molinst_pro_test

for task in protein_protein_function protein_general_function protein_catalytic_activity protein_domain_motif
do
        echo ${task}
        python main.py \
                predict_only=true \
                task=ft_molinst \
                test_task=molinst_pro_understand \
                model.name=google/t5-v1_1-base \
                model.checkpoint_path=${ckpt_path} \
                optim.batch_size=64 optim.grad_acc=1 \
                data.data_dir=${data_dir}/${task} \
                data.task_dir=${task_dir} \
                data.max_seq_len=512 data.max_target_len=256 \
                hydra.run.dir=${run_dir}/${task}
done

task=protein_protein_design
echo ${task}
python main.py \
        predict_only=true \
        task=ft_molinst \
        test_task=molinst_pro_understand \
        model.name=google/t5-v1_1-base \
        model.checkpoint_path=${ckpt_path} \
        optim.batch_size=64 optim.grad_acc=1 \
        data.data_dir=${data_dir}/${task} \
        data.task_dir=${task_dir} \
        data.max_seq_len=512 data.max_target_len=256 \
        hydra.run.dir=${run_dir}/${task}