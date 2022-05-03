#!/bin/bash

#echo $LD_LIBRARY_PATH
#export PATH=/usr/local/cuda-9.0/bin/:$PATH
#export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64/:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/home/rahulee16/anaconda3/pkgs/cudatoolkit-10.0.130-0/lib/

datapaths='../../data'
ridleyPaths='../../data'
word_feat_path=$datapaths'/WordFeatures.txt'
sent_feat_path=$datapaths'/SentenceFeatures.txt' 
essay_feat_path='../../data/EssayFeatures.txt'
ridley_feat_path="$ridleyPaths/hand_crafted_v3.csv $ridleyPaths/allreadability.pickle" 
bert_repr_path="../../../../Models/bert"
word_feat_flag=0
sent_feat_flag=0
essay_feat_flag=0
ridley_feat_flag=0
batch_size=16
loss='regression_and_ranking'


num_epochs=30
gpu=1
trait_id=6

for prompt_id in 1
do
	for fold_id in 0
	do
    	echo "Started Prompt " $prompt_id 
	echo "Fold " $fold_id
	#CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREAD=5 python ATTN_BERT.py --word_feat_flag $word_feat_flag --sent_feat_flag $sent_feat_flag --essay_feat_flag $essay_feat_flag --ridley_feat_flag $ridley_feat_flag --word_feat_path $word_feat_path --sent_feat_path $sent_feat_path --essay_feat_path $essay_feat_path --ridley_feat_path $ridley_feat_path --bert_repr_path $bert_repr_path --loss 'regression' -wt model_weights --checkpoint_path checkpoint_path --prompt_filePath ../../attn-git/ASAP-Essay-Traits-Scoring/data/$prompt_id.words.txt  --train ../../attn-git/ASAP-Essay-Traits-Scoring/data/CD-AEG/Prompt$prompt_id/fold_$fold_id/train.BERT.tsv --dev ../../attn-git/ASAP-Essay-Traits-Scoring/data/CD-AEG/Prompt$prompt_id/fold_$fold_id/dev.BERT.tsv --test ../../attn-git/ASAP-Essay-Traits-Scoring/data/CD-AEG/Prompt$prompt_id/fold_$fold_id/test.BERT.tsv -o out_dir --num_epochs ${num_epochs} --attr_column $TRAIT --attr_loss_weights $WT --batch_size 16 --prompt_id $prompt_id #> ../logs_self_attn/ALBERT/CD_AEG/Prompt-$prompt_id-Fold-$fold_id.txt
	CUDA_VISIBLE_DEVICES=$gpu python ATTN_BERT_STL.py --word_feat_flag $word_feat_flag --sent_feat_flag $sent_feat_flag --essay_feat_flag $essay_feat_flag --ridley_feat_flag $ridley_feat_flag --word_feat_path $word_feat_path --sent_feat_path $sent_feat_path --essay_feat_path $essay_feat_path --ridley_feat_path $ridley_feat_path --bert_repr_path $bert_repr_path --prompt_filePath ../../../data/$prompt_id.words.txt --loss $loss --checkpoint_path checkpoint_path --train ../../data/fold_$fold_id/train.tsv --dev ../../data/fold_$fold_id/dev.tsv --test ../../data/fold_$fold_id/test.tsv -o out_dir --num_epochs ${num_epochs} --score_index $trait_id --batch_size $batch_size --prompt_id $prompt_id #> ../../../logs/STL_Only/AEG/BERT/Prompt-$prompt_id-FT-Fold-$fold_id.txt

	done
	echo "Finished prompt id: " $prompt_id
done
