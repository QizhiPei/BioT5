ckpt_path=path_to_your_model
data_dir=../data/splits_plus/molinst/eval
task_dir=../data/tasks_plus
run_dir=../logs/molinst_mol_test

for task in molecule_description_guided_molecule_design molecule_reagent_prediction molecule_forward_reaction_prediction molecule_retrosynthesis
do
        echo ${task}
        python main.py \
                predict_only=true \
                task=ft_molinst \
                test_task=molinst_mol_gen \
                model.name=google/t5-v1_1-base \
                model.checkpoint_path=${ckpt_path} \
                optim.batch_size=64 optim.grad_acc=1 \
                data.task_dir=${task_dir} \
                data.data_dir=${data_dir}/${task} \
                data.max_seq_len=512 data.max_target_len=256 \
                generation_config.repetition_penalty=1.1 \
                hydra.run.dir=${run_dir}/${task}
done

echo molinst_mol_understand
python main.py \
        predict_only=true \
        task=ft_molinst \
        test_task=molinst_mol_understand \
        model.name=google/t5-v1_1-base \
        model.checkpoint_path=${ckpt_path} \
        optim.batch_size=64 optim.grad_acc=1 \
        data.task_dir=${task_dir} \
        data.data_dir=${data_dir}/molecule_molecular_description_generation \
        data.max_seq_len=512 data.max_target_len=256 \
        generation_config.repetition_penalty=1.1 \
        hydra.run.dir=${run_dir}/molinst_mol_understand

echo molinst_mol_property_prediction
python main.py \
        predict_only=true \
        task=ft_molinst \
        test_task=molinst_mol_property_prediction \
        model.name=google/t5-v1_1-base \
        model.checkpoint_path=${ckpt_path} \
        optim.batch_size=64 optim.grad_acc=1 \
        data.task_dir=${task_dir} \
        data.data_dir=${data_dir}/molecule_property_prediction \
        data.max_seq_len=512 data.max_target_len=256 \
        hydra.run.dir=${run_dir}/molinst_mol_property_prediction