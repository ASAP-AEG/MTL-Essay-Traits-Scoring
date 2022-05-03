#!/bin/bash

# $LD_LIBRARY_PATH
#For bertENV use cuda-11.0 and for newPython use cuda-9.0
#export PATH=/usr/local/cuda-9.0/bin/:$PATH
#export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64/:$LD_LIBRARY_PATH

#echo $PATH
#echo $LD_LIBRARY_PATH
#OMP_NUM_THREADS=5

num_epochs=100
gpu=3

#prompt_id=3
datapaths='../../data'
word_feat=$datapaths'/NormalizedWordFeatures.txt'
sent_feat=$datapaths'/NormalizedSentenceFeatures.txt' 
essay_feat=$datapaths'/NormalizedEssayFeatures.txt'
hand_feat_path=$datapaths'/hand_crafted_v3.csv'
readability_path=$datapaths'/allreadability.pickle'
logFolder = './logs'

AX_TASK='6 8 9 10 11'
PR_TASK='7'
AX_WT='1 1 1 1 1'
PR_WT='1'

for prompt_id in 3 4 5 7
do
	for fold_id in 0 1 2 3 4
	do
		echo "Started Prompt " $prompt_id 
		echo "Fold " $fold_id

		AX_TASK='7 8 9 10'
		PR_TASK='6'
		AX_WT='1 1 1 1'
		PR_WT='1'
		# CUDA_VISIBLE_DEVICES=$gpu python ATTN.py --embedding_dict $datapaths'/glove/glove.6B.50d.txt.gz' --word_feat $word_feat --sent_feat $sent_feat --essay_feat $essay_feat --hand_feat $hand_feat_path --readability_feat $readability_path --prompt_filePath $datapaths/$prompt_id.words.txt -wt model_weights --checkpoint_path checkpoint_path --train ../../data/CD-AEG/Prompt$prompt_id/fold_$fold_id/train.tsv --dev ../../data/CD-AEG/Prompt$prompt_id/fold_$fold_id/dev.tsv --test ../../data/CD-AEG/Prompt$prompt_id/fold_$fold_id/test.tsv -o out_dir --num_epochs ${num_epochs} --attr_column $TRAIT --attr_loss_weights $WT --batch_size 100 --prompt_id $prompt_id > ./logs/MTL_CD_AEG/Prompt-$prompt_id-Fold-$fold_id.txt
		CUDA_VISIBLE_DEVICES=$gpu python ATTN.py --embedding_dict $datapaths'/glove.6B.50d.txt.gz' --word_feat $word_feat --sent_feat $sent_feat --essay_feat $essay_feat --hand_feat $hand_feat_path --readability_feat $readability_path --prompt_filePath $datapaths/$prompt_id.words.txt -wt model_weights --checkpoint_path checkpoint_path --train ../../data/fold_$fold_id/train.tsv --dev ../../data/fold_$fold_id/dev.tsv --test ../../data/fold_$fold_id/test.tsv -o out_dir --num_epochs ${num_epochs} --primary_task_column $PR_TASK --primary_task_weight $PR_WT --auxiliary_task_column $AX_TASK --auxiliary_task_weight $AX_WT --batch_size 100 --prompt_id $prompt_id > ./logs/MTL-Prompt-$prompt_id-Task-$PR_TASK-Fold-$fold_id.txt

		# AX_TASK='6 7 9 10 11'
		# PR_TASK='8'
		# AX_WT='1 1 1 1 1'
		# PR_WT='1'
		# # CUDA_VISIBLE_DEVICES=$gpu python ATTN.py --embedding_dict $datapaths'/glove/glove.6B.50d.txt.gz' --word_feat $word_feat --sent_feat $sent_feat --essay_feat $essay_feat --hand_feat $hand_feat_path --readability_feat $readability_path --prompt_filePath $datapaths/$prompt_id.words.txt -wt model_weights --checkpoint_path checkpoint_path --train ../../data/CD-AEG/Prompt$prompt_id/fold_$fold_id/train.tsv --dev ../../data/CD-AEG/Prompt$prompt_id/fold_$fold_id/dev.tsv --test ../../data/CD-AEG/Prompt$prompt_id/fold_$fold_id/test.tsv -o out_dir --num_epochs ${num_epochs} --attr_column $TRAIT --attr_loss_weights $WT --batch_size 100 --prompt_id $prompt_id > ./logs/MTL_CD_AEG/Prompt-$prompt_id-Fold-$fold_id.txt
		# CUDA_VISIBLE_DEVICES=$gpu python ATTN.py --embedding_dict $datapaths'/glove/glove.6B.50d.txt.gz' --word_feat $word_feat --sent_feat $sent_feat --essay_feat $essay_feat --hand_feat $hand_feat_path --readability_feat $readability_path --prompt_filePath $datapaths/$prompt_id.words.txt -wt model_weights --checkpoint_path checkpoint_path --train ../../data/fold_$fold_id/train.tsv --dev ../../data/fold_$fold_id/dev.tsv --test ../../data/fold_$fold_id/test.tsv -o out_dir --num_epochs ${num_epochs} --primary_task_column $PR_TASK --primary_task_weight $PR_WT --auxiliary_task_column $AX_TASK --auxiliary_task_weight $AX_WT --batch_size 100 --prompt_id $prompt_id > $logFolder/MTL-Prompt-$prompt_id-Task-$PR_TASK-Fold-$fold_id.txt

		# AX_TASK='6 7 8 10 11'
		# PR_TASK='9'
		# AX_WT='1 1 1 1 1'
		# PR_WT='1'
		# # CUDA_VISIBLE_DEVICES=$gpu python ATTN.py --embedding_dict $datapaths'/glove/glove.6B.50d.txt.gz' --word_feat $word_feat --sent_feat $sent_feat --essay_feat $essay_feat --hand_feat $hand_feat_path --readability_feat $readability_path --prompt_filePath $datapaths/$prompt_id.words.txt -wt model_weights --checkpoint_path checkpoint_path --train ../../data/CD-AEG/Prompt$prompt_id/fold_$fold_id/train.tsv --dev ../../data/CD-AEG/Prompt$prompt_id/fold_$fold_id/dev.tsv --test ../../data/CD-AEG/Prompt$prompt_id/fold_$fold_id/test.tsv -o out_dir --num_epochs ${num_epochs} --attr_column $TRAIT --attr_loss_weights $WT --batch_size 100 --prompt_id $prompt_id > ./logs/MTL_CD_AEG/Prompt-$prompt_id-Fold-$fold_id.txt
		# CUDA_VISIBLE_DEVICES=$gpu python ATTN.py --embedding_dict $datapaths'/glove/glove.6B.50d.txt.gz' --word_feat $word_feat --sent_feat $sent_feat --essay_feat $essay_feat --hand_feat $hand_feat_path --readability_feat $readability_path --prompt_filePath $datapaths/$prompt_id.words.txt -wt model_weights --checkpoint_path checkpoint_path --train ../../data/fold_$fold_id/train.tsv --dev ../../data/fold_$fold_id/dev.tsv --test ../../data/fold_$fold_id/test.tsv -o out_dir --num_epochs ${num_epochs} --primary_task_column $PR_TASK --primary_task_weight $PR_WT --auxiliary_task_column $AX_TASK --auxiliary_task_weight $AX_WT --batch_size 100 --prompt_id $prompt_id > $logFolder/MTL-Prompt-$prompt_id-Task-$PR_TASK-Fold-$fold_id.txt

		# AX_TASK='6 7 8 9 11'
		# PR_TASK='10'
		# AX_WT='1 1 1 1 1'
		# PR_WT='1'
		# # CUDA_VISIBLE_DEVICES=$gpu python ATTN.py --embedding_dict $datapaths'/glove/glove.6B.50d.txt.gz' --word_feat $word_feat --sent_feat $sent_feat --essay_feat $essay_feat --hand_feat $hand_feat_path --readability_feat $readability_path --prompt_filePath $datapaths/$prompt_id.words.txt -wt model_weights --checkpoint_path checkpoint_path --train ../../data/CD-AEG/Prompt$prompt_id/fold_$fold_id/train.tsv --dev ../../data/CD-AEG/Prompt$prompt_id/fold_$fold_id/dev.tsv --test ../../data/CD-AEG/Prompt$prompt_id/fold_$fold_id/test.tsv -o out_dir --num_epochs ${num_epochs} --attr_column $TRAIT --attr_loss_weights $WT --batch_size 100 --prompt_id $prompt_id > ./logs/MTL_CD_AEG/Prompt-$prompt_id-Fold-$fold_id.txt
		# CUDA_VISIBLE_DEVICES=$gpu python ATTN.py --embedding_dict $datapaths'/glove/glove.6B.50d.txt.gz' --word_feat $word_feat --sent_feat $sent_feat --essay_feat $essay_feat --hand_feat $hand_feat_path --readability_feat $readability_path --prompt_filePath $datapaths/$prompt_id.words.txt -wt model_weights --checkpoint_path checkpoint_path --train ../../data/fold_$fold_id/train.tsv --dev ../../data/fold_$fold_id/dev.tsv --test ../../data/fold_$fold_id/test.tsv -o out_dir --num_epochs ${num_epochs} --primary_task_column $PR_TASK --primary_task_weight $PR_WT --auxiliary_task_column $AX_TASK --auxiliary_task_weight $AX_WT --batch_size 100 --prompt_id $prompt_id > $logFolder/MTL-Prompt-$prompt_id-Task-$PR_TASK-Fold-$fold_id.txt

		# AX_TASK='6 7 8 9 10'
		# PR_TASK='11'
		# AX_WT='1 1 1 1 1'
		# PR_WT='1'
		# # CUDA_VISIBLE_DEVICES=$gpu python ATTN.py --embedding_dict $datapaths'/glove/glove.6B.50d.txt.gz' --word_feat $word_feat --sent_feat $sent_feat --essay_feat $essay_feat --hand_feat $hand_feat_path --readability_feat $readability_path --prompt_filePath $datapaths/$prompt_id.words.txt -wt model_weights --checkpoint_path checkpoint_path --train ../../data/CD-AEG/Prompt$prompt_id/fold_$fold_id/train.tsv --dev ../../data/CD-AEG/Prompt$prompt_id/fold_$fold_id/dev.tsv --test ../../data/CD-AEG/Prompt$prompt_id/fold_$fold_id/test.tsv -o out_dir --num_epochs ${num_epochs} --attr_column $TRAIT --attr_loss_weights $WT --batch_size 100 --prompt_id $prompt_id > ./logs/MTL_CD_AEG/Prompt-$prompt_id-Fold-$fold_id.txt
		# CUDA_VISIBLE_DEVICES=$gpu python ATTN.py --embedding_dict $datapaths'/glove/glove.6B.50d.txt.gz' --word_feat $word_feat --sent_feat $sent_feat --essay_feat $essay_feat --hand_feat $hand_feat_path --readability_feat $readability_path --prompt_filePath $datapaths/$prompt_id.words.txt -wt model_weights --checkpoint_path checkpoint_path --train ../../data/fold_$fold_id/train.tsv --dev ../../data/fold_$fold_id/dev.tsv --test ../../data/fold_$fold_id/test.tsv -o out_dir --num_epochs ${num_epochs} --primary_task_column $PR_TASK --primary_task_weight $PR_WT --auxiliary_task_column $AX_TASK --auxiliary_task_weight $AX_WT --batch_size 100 --prompt_id $prompt_id > $logFolder/MTL-Prompt-$prompt_id-Task-$PR_TASK-Fold-$fold_id.txt

	done
	echo "Finished prompt id: " $prompt_id
done