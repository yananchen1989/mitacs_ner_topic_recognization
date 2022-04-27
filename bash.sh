


CUDA_VISIBLE_DEVICES=3 python -u /home/w/wluyliu/yananc/topic_classification_augmentation/run_summarization_no_trainer.py \
            --num_train_epochs 3 \
            --dataset_name "c4" \
            --para 'ss' \
            --model_name_or_path  facebook/bart-base \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 16 \
            --output_dir '/scratch/w/wluyliu/yananc/finetunes/finetunes/bart_c4_ss' \
            --max_target_length 128 \
            --max_source_length 128 \
            --val_max_target_length 128 \
            --preprocessing_num_workers 32 --overwrite_cache True \
            --text_column text1 \
            --summary_column text2 \
            --max_length 128 \
            --model_type bart  --local_files_only




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


