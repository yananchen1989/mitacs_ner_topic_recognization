


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



CUDA_VISIBLE_DEVICES=3 python -u /home/w/wluyliu/yananc/nlp4quantumpapers/run_summarization_no_trainer.py \
            --num_train_epochs 3 \
            --model_name_or_path  t5-base \
            --per_device_train_batch_size 32   --per_device_eval_batch_size 8 \
            --output_dir '/scratch/w/wluyliu/yananc/finetunes/t5_nerd_test' \
            --max_target_length 128 \
            --max_source_length 128 \
            --val_max_target_length 128 \
            --overwrite_cache True \
            --text_column text1 \
            --summary_column text2 \
            --debug_cnt 20480  \
            --model_type t5  --local_files_only --tags_column tags_coarse



sbatch submit_t5_nerd.slurm 1024 tags_coarse



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


