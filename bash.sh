

for gpu in 0 1 2 3 
do
CUDA_VISIBLE_DEVICES=${gpu} nohup  python -u /home/w/wluyliu/yananc/nlp4quantumpapers/run_ner_no_trainer.py \
          --dataset_name "tqi" \
          --model_name_or_path roberta-large \
          --dataset_config_name "supervised" \
          --output_dir '/scratch/w/wluyliu/yananc/finetunes/roberta_tqi' \
          --text_column_name "tokens" \
          --label_column_name "tags" \
          --num_train_epochs 20 \
          --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
          --debug_cnt  16 \
          --local_files_only  > bert_tagger_tqi_16_${gpu}.log & 
done


CUDA_VISIBLE_DEVICES=1  python -u /home/w/wluyliu/yananc/nlp4quantumpapers/run_ner_no_trainer.py \
          --dataset_name "few_nerd_local" \
          --model_name_or_path roberta-large \
          --dataset_config_name "supervised" \
          --output_dir '/scratch/w/wluyliu/yananc/finetunes/roberta_fewnerd' \
          --text_column_name "tokens" \
          --label_column_name "tags_coarse" \
          --num_train_epochs 20 \
          --per_device_train_batch_size 32 --per_device_eval_batch_size 32 \
          --debug_cnt  -1 \
          --local_files_only --da 1 --da_ver "fewnerd_SIS_1"





CUDA_VISIBLE_DEVICES=3 python -u unit_test.py 


          





CUDA_VISIBLE_DEVICES=2 python -u /home/w/wluyliu/yananc/nlp4quantumpapers/run_summarization_no_trainer.py \
            --num_train_epochs 12 \
            --model_name_or_path  t5-large \
            --per_device_train_batch_size 8   --per_device_eval_batch_size 256 \
            --output_dir '/scratch/w/wluyliu/yananc/finetunes/t5_nerd_test' \
            --max_target_length 128 \
            --max_source_length 128 \
            --val_max_target_length 128 \
            --overwrite_cache True \
            --text_column text1 \
            --summary_column text2 \
            --debug_cnt 2048  \
            --model_type t5  --local_files_only --tags_column tags_fine



sbatch submit_t5_nerd.slurm 1024 tags_fine;
sbatch submit_t5_nerd.slurm 2048 tags_fine;
sbatch submit_t5_nerd.slurm -1   tags_fine;


for i in 1 2 3 4 
do
    for da in 0 1 
    do
        for samplecnt in 1024 2048 -1 
        do
        for da_ver in fewnerd_both_SIS_SR_0.1.1 fewnerd_both_SIS_SR_0.3 fewnerd_both_SIS_SR_0.5 \
                        fewnerd_SIS_0.1.1 fewnerd_SIS_0.3 fewnerd_SIS_0.5 fewnerd_SIS_0.7.7 fewnerd_SIS_1 \
                        fewnerd_SR_0.1.1 fewnerd_SR_0.3 fewnerd_SR_0.5 fewnerd_SR_0.7.7 fewnerd_SR_1 
            do 
                sbatch submit_roberta_nerd.slurm ${samplecnt} ${da_ver} ${da};
            done
        done 
    done
done 


sbatch submit_t5_nerd_da.slurm -1 0.8;
sbatch submit_t5_nerd_da.slurm -1 0.5;
sbatch submit_t5_nerd_da.slurm -1 0.3;
sbatch submit_t5_nerd_da.slurm -1 0.15;




sbatch test.slurm 0.8;
sbatch test.slurm 0.5;
sbatch test.slurm 0.3;
sbatch test.slurm 0.15;



for samplecnt in  -1
do
    for p in 0.8 0.5 0.3 0.15
    do
    sbatch submit_roberta_nerd.slurm ${samplecnt} 1 ${p};
    sbatch submit_roberta_nerd.slurm ${samplecnt} 0 ${p};
    done
done





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


