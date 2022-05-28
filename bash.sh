


CUDA_VISIBLE_DEVICES=0  python -u /home/w/wluyliu/yananc/nlp4quantumpapers/run_ner_no_trainer.py \
          --dataset_name "tqi" \
          --model_name_or_path roberta-large \
          --dataset_config_name "supervised" \
          --output_dir '/scratch/w/wluyliu/yananc/finetunes/roberta_tqi' \
          --text_column_name "tokens" \
          --label_column_name "tags" \
          --num_train_epochs 12 \
          --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
          --debug_cnt  -1 \
          --local_files_only



CUDA_VISIBLE_DEVICES=0 python -u /home/w/wluyliu/yananc/nlp4quantumpapers/run_summarization_no_trainer.py \
            --num_train_epochs 12 \
            --model_name_or_path  t5-large \
            --per_device_train_batch_size 8   --per_device_eval_batch_size 512 \
            --output_dir '/scratch/w/wluyliu/yananc/finetunes/t5_nerd_test' \
            --max_target_length 128 \
            --max_source_length 128 \
            --val_max_target_length 128 \
            --overwrite_cache True \
            --text_column text1 \
            --summary_column text2 \
            --debug_cnt 1024  \
            --model_type t5  --local_files_only --tags_column tags_coarse



sbatch submit_t5_nerd.slurm 1024 tags_coarse;
sbatch submit_t5_nerd.slurm 2048 tags_coarse;
sbatch submit_t5_nerd.slurm -1   tags_coarse;



# t5-base
0 tags_coarse 2048 {'precision': 0.4344045480566138, 'recall': 0.30484772505315944, 'f1': 0.35827346913856606}

0 tags_coarse 1024 {'precision': 0.0007484482602879376, 'recall': 0.0012051781100867866, 'f1': 0.0009234247382303149}





# t5-large 
0 tags_coarse 2048 {'precision': 0.5266319243389307, 'recall': 0.47245752439793043, 'f1': 0.4980759541733905}





############################  
module load anaconda3;source activate env

/home/w/wluyliu/yananc/topic_classification_augmentation/

/scratch/w/wluyliu/yananc

conda create -n env python=3.8


conda install -c /scinet/mist/ibm/open-ce tensorflow==2.7.0 cudatoolkit=11.2
conda install -c /scinet/mist/ibm/open-ce scikit-learn
conda install -c /scinet/mist/ibm/open-ce pytorch=1.10.1 cudatoolkit=11.2
conda install -c /scinet/mist/ibm/open-ce transformers==4.9.2
conda install -c /scinet/mist/ibm/open-ce tensorflow-text
conda install -c /scinet/mist/ibm/open-ce tensorflow_hub
conda install -c /scinet/mist/ibm/open-ce matplotlib

conda env remove --name myenv



cd $SCRATCH

squeue --me

scancel -i JOBID #cancels a specific job.
sacct #gives information about your recent jobs.
sinfo -p #compute gives a list of available nodes.
qsum #gives a summary of the queue by user.


