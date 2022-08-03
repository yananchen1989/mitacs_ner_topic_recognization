



# for TQI
CUDA_VISIBLE_DEVICES=3   python -u /home/w/wluyliu/yananc/nlp4quantumpapers/run_ner_no_trainer.py \
          --dataset_name "tqi" \
          --model_name_or_path roberta-large \
          --dataset_config_name "supervised" \
          --output_dir '/scratch/w/wluyliu/yananc/finetunes/roberta_tqi' \
          --text_column_name "tokens" \
          --label_column_name "tags" \
          --num_train_epochs 4 \
          --per_device_train_batch_size 16 --per_device_eval_batch_size 16 \
          --local_files_only















CUDA_VISIBLE_DEVICES=3 python -u /home/w/wluyliu/yananc/nlp4quantumpapers/run_clm_no_trainer.py \
        --num_train_epochs 25 \
        --dataset_name 'fewnerd' \
        --model_name_or_path gpt2 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --output_dir /scratch/w/wluyliu/yananc/finetunes/gpt2_fewnerd \
        --preprocessing_num_workers 128 --overwrite_cache True --block_size 256 \
         --debug_cnt -1
          


for k in 1024 2048 -1 
do 
    sbatch test.slurm ${k}
done








CUDA_VISIBLE_DEVICES=2 python -u /home/w/wluyliu/yananc/nlp4quantumpapers/run_summarization_no_trainer.py \
            --num_train_epochs 7 \
            --model_name_or_path  t5-base \
            --per_device_train_batch_size 16   --per_device_eval_batch_size 128 \
            --output_dir '/scratch/w/wluyliu/yananc/finetunes/t5_nerd_test' \
            --max_target_length 256 \
            --max_source_length 256 \
            --val_max_target_length 256 \
            --overwrite_cache True \
            --text_column text1 \
            --summary_column text2 \
            --debug_cnt 10240  \
            --model_type t5  --local_files_only --tags_column tags_coarse --seed 1 --binomial 0.5



sbatch submit_t5_nerd.slurm 1024 tags_fine;
sbatch submit_t5_nerd.slurm 2048 tags_fine;
sbatch submit_t5_nerd.slurm -1   tags_fine;



for da in 1 0
do
    for samplecnt in 1024 2048 
    do
    for da_ver in fewnerd_both_SIS_SR_0.1.1 fewnerd_both_SIS_SR_0.3 fewnerd_both_SIS_SR_0.5 \
                    fewnerd_SIS_0.1.1 fewnerd_SIS_0.3 fewnerd_SIS_0.5 fewnerd_SIS_0.7.7 fewnerd_SIS_1 \
                    fewnerd_SR_0.1.1 fewnerd_SR_0.3 fewnerd_SR_0.5 fewnerd_SR_0.7.7 fewnerd_SR_1 
        do 
            sbatch submit_roberta_nerd.slurm ${samplecnt} ${da_ver} ${da};
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


