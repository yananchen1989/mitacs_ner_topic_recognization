CUDA_VISIBLE_DEVICES=3 python -u /home/w/wluyliu/yananc/nlp4quantumpapers/run_summarization_no_trainer.py \
            --num_train_epochs 7 \
            --model_name_or_path  t5-large \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 8 \
            --output_dir '/scratch/w/wluyliu/yananc/finetunes/t5_nerd' \
            --max_target_length 128 \
            --max_source_length 128 \
            --val_max_target_length 128 \
             --overwrite_cache True \
            --text_column text1 \
            --summary_column text2 \
            --debug_cnt -1 \
            --model_type t5  --local_files_only --tags_column "tags_fine"




CUDA_VISIBLE_DEVICES=2  python -u /home/w/wluyliu/yananc/nlp4quantumpapers/run_ner_no_trainer.py \
          --dataset_name "few_nerd_local" \
          --model_name_or_path roberta-large \
          --dataset_config_name "supervised" \
          --output_dir /scratch/w/wluyliu/yananc/finetunes/roberta_nerd_fine \
          --text_column_name "tokens" \
          --label_column_name "tags_fine" \
          --num_train_epochs 7 \
          --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
           --local_files_only




# fine: 'precision': 0.676235202035026, 'recall': 0.7096595342724548, 'f1': 0.6925443122952216
# coarse: 


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


