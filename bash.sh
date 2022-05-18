
curl https://api-inference.huggingface.co/models/xlm-roberta-large-finetuned-conll03-english \
    -X POST \
    -d '{"inputs": "Conventional computer “bits” have a value of either 0 or 1, but quantum bits"}' \
    -H "Authorization: Bearer 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"






CUDA_VISIBLE_DEVICES=0 python -u /home/w/wluyliu/yananc/topic_classification_augmentation/run_summarization_no_trainer.py \
            --num_train_epochs 7 \
            --train_file "/scratch/w/wluyliu/yananc/df_nerd_train.csv" \
            --validation_file  "/scratch/w/wluyliu/yananc/df_nerd_test.csv" \
            --model_name_or_path  t5-base \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --output_dir '/scratch/w/wluyliu/yananc/finetunes/t5_nerd' \
            --max_target_length 16 \
            --max_source_length 64 \
            --val_max_target_length 16 \
            --preprocessing_num_workers 32 --overwrite_cache True \
            --text_column text1 \
            --summary_column text2 \
            --max_length 64 \
            --model_type t5  --local_files_only

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


