

CUDA_VISIBLE_DEVICES=0  python -u run_mlm_no_trainer.py \
    --num_train_epochs 7 \
    --train_file '/home/w/wluyliu/yananc/nlp4quantumpapers/finetune/df_arxiv.train.txt' \
    --validation_file '/home/w/wluyliu/yananc/nlp4quantumpapers/finetune/df_arxiv.test.txt' \
    --model_name_or_path "allenai/scibert_scivocab_uncased" \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --output_dir '/scratch/w/wluyliu/yananc/finetune/arxiv_scibert' \
    --preprocessing_num_workers 128 --overwrite_cache True \
    --mlm_probability 0.15 \
    --max_seq_length 256 \
    --use_slow_tokenizer > /scratch/w/wluyliu/yananc/arxiv_scibert.log 











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

