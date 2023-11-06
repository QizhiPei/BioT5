[ -z "${task}" ] && task="dti"
[ -z "${result_file_path}" ] && result_file_path="tmp.tsv"
[ -z "${model_path}" ] && model_path="path_to_your_model"
[ -z "${log_path}" ] && log_path="path_to_your_log"
[ -z "${task_dir}" ] && task_dir="../data/tasks"
[ -z "${data_dir}" ] && data_dir="../data/splits/dti/dti_human"

cd biot5

python main.py \
        task=${task} \
        test_task=${task} \
        result_fn=${result_file_path} \
        model.checkpoint_path=${model_path} \
        data.task_dir=${task_dir} \
        data.data_dir=${data_dir} \
        hydra.run.dir=${log_path} \
        predict_only=true \
        optim.grad_acc=1