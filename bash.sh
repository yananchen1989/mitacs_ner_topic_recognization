


CUDA_VISIBLE_DEVICES=0  python -u /home/w/wluyliu/yananc/nlp4quantumpapers/run_ner_no_trainer.py \
          --dataset_name "few_nerd_local" \
          --model_name_or_path roberta-large \
          --dataset_config_name "supervised" \
          --output_dir '/scratch/w/wluyliu/yananc/finetunes/roberta_nerd_test' \
          --text_column_name "tokens" \
          --label_column_name "tags_coarse" \
          --num_train_epochs 12 \
          --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
          --debug_cnt  -1 \
          --local_files_only




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


