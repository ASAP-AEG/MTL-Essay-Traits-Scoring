#!/bin/bash

# echo $LD_LIBRARY_PATH
export PATH=/usr/local/cuda-9.0/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64/:$LD_LIBRARY_PATH

num_epochs=100
gpu=3
datapaths='../glove'
ridleyPaths='../cross-prompt-attribute-aes/data'
word_feat_path=$datapaths'/WordFeatures.txt'
sent_feat_path=$datapaths'/SentenceFeatures.txt' 
essay_feat_path='../data/EssayFeatures.txt'
ridley_feat_path="$ridleyPaths/hand_crafted_v3.csv $ridleyPaths/allreadability.pickle" 
word_feat_flag=0
sent_feat_flag=0
essay_feat_flag=1
ridley_feat_flag=1

for prompt_id in 1 2
do
	echo "Started Prompt ID " $prompt_id
	for fold_id in 0 1 2 3 4
	do
	    echo "Started Fold " $fold_id
		for trait_id in 6 7 8 9 10 11
		do
			echo "Started Trait " $trait_id
		    #CUDA_VISIBLE_DEVICES=$gpu  python ATTN_STL.py --rnn_type 'Bi-LSTM' --word_feat_flag $word_feat_flag --sent_feat_flag $sent_feat_flag --essay_feat_flag $essay_feat_flag --ridley_feat_flag $ridley_feat_flag --word_feat_path $word_feat_path --sent_feat_path $sent_feat_path --essay_feat_path $essay_feat_path --ridley_feat_path $ridley_feat_path  --train ./data/fold_$fold_id/train.tsv --dev ./data/fold_$fold_id/dev.tsv --test ./data/fold_$fold_id/test.tsv -o out_dir --num_epochs ${num_epochs} --batch_size 100 --prompt_id $prompt_id #> ./logs_self_attn/Bi_LSTM_Without_TRAITS/Prompt-$prompt_id/Fold-$fold_id.txt
		CUDA_VISIBLE_DEVICES=$gpu  python ATTN_STL.py --rnn_type 'Bi-LSTM' --word_feat_flag $word_feat_flag --sent_feat_flag $sent_feat_flag --essay_feat_flag $essay_feat_flag --ridley_feat_flag $ridley_feat_flag --word_feat_path $word_feat_path --sent_feat_path $sent_feat_path --essay_feat_path $essay_feat_path --ridley_feat_path $ridley_feat_path --prompt_filePath ../../attn-git/ASAP-Essay-Traits-Scoring/data/$prompt_id.words.txt --train ../../attn-git/ASAP-Essay-Traits-Scoring/data/CD-AEG/Prompt$prompt_id/fold_$fold_id/train.tsv --dev ../../attn-git/ASAP-Essay-Traits-Scoring/data/CD-AEG/Prompt$prompt_id/fold_$fold_id/dev.tsv --test ../../attn-git/ASAP-Essay-Traits-Scoring/data/CD-AEG/Prompt$prompt_id/fold_$fold_id/test.tsv -o out_dir --num_epochs ${num_epochs} --batch_size 100 --prompt_id $prompt_id --score_index $trait_id > ../logs_self_attn/STL/CD-AEG/Prompt-$prompt_id-Fold-$fold_id-Trait-$trait_id.txt
		done
		echo "Finished Trait: " $trait_id
	done
	echo "Finished fold id: " $fold_id
done
echo "Finished prompt id: " $prompt_id

for prompt_id in 3 4 5 6 7
do
	echo "Started Prompt ID " $prompt_id
	for fold_id in 0 1 2 3 4
	do
	    echo "Started Fold " $fold_id
		for trait_id in 6 7 8 9 10
		do
			echo "Started Trait " $trait_id
		    #CUDA_VISIBLE_DEVICES=$gpu  python ATTN_STL.py --rnn_type 'Bi-LSTM' --word_feat_flag $word_feat_flag --sent_feat_flag $sent_feat_flag --essay_feat_flag $essay_feat_flag --ridley_feat_flag $ridley_feat_flag --word_feat_path $word_feat_path --sent_feat_path $sent_feat_path --essay_feat_path $essay_feat_path --ridley_feat_path $ridley_feat_path  --train ./data/fold_$fold_id/train.tsv --dev ./data/fold_$fold_id/dev.tsv --test ./data/fold_$fold_id/test.tsv -o out_dir --num_epochs ${num_epochs} --batch_size 100 --prompt_id $prompt_id #> ./logs_self_attn/Bi_LSTM_Without_TRAITS/Prompt-$prompt_id/Fold-$fold_id.txt
		CUDA_VISIBLE_DEVICES=$gpu  python ATTN_STL.py --rnn_type 'Bi-LSTM' --word_feat_flag $word_feat_flag --sent_feat_flag $sent_feat_flag --essay_feat_flag $essay_feat_flag --ridley_feat_flag $ridley_feat_flag --word_feat_path $word_feat_path --sent_feat_path $sent_feat_path --essay_feat_path $essay_feat_path --ridley_feat_path $ridley_feat_path --prompt_filePath ../../attn-git/ASAP-Essay-Traits-Scoring/data/$prompt_id.words.txt --train ../../attn-git/ASAP-Essay-Traits-Scoring/data/CD-AEG/Prompt$prompt_id/fold_$fold_id/train.tsv --dev ../../attn-git/ASAP-Essay-Traits-Scoring/data/CD-AEG/Prompt$prompt_id/fold_$fold_id/dev.tsv --test ../../attn-git/ASAP-Essay-Traits-Scoring/data/CD-AEG/Prompt$prompt_id/fold_$fold_id/test.tsv -o out_dir --num_epochs ${num_epochs} --batch_size 100 --prompt_id $prompt_id --score_index $trait_id > ../logs_self_attn/STL/CD-AEG/Prompt-$prompt_id-Fold-$fold_id-Trait-$trait_id.txt
		done
		echo "Finished Trait: " $trait_id
	done
	echo "Finished fold id: " $fold_id
done
echo "Finished prompt id: " $prompt_id

for prompt_id in 8
do
	echo "Started Prompt ID " $prompt_id
	for fold_id in 0 1 2 3 4
	do
	    echo "Started Fold " $fold_id
		for trait_id in 6 7 8 9 10 11 12
		do
			echo "Started Trait " $trait_id
		    #CUDA_VISIBLE_DEVICES=$gpu  python ATTN_STL.py --rnn_type 'Bi-LSTM' --word_feat_flag $word_feat_flag --sent_feat_flag $sent_feat_flag --essay_feat_flag $essay_feat_flag --ridley_feat_flag $ridley_feat_flag --word_feat_path $word_feat_path --sent_feat_path $sent_feat_path --essay_feat_path $essay_feat_path --ridley_feat_path $ridley_feat_path  --train ./data/fold_$fold_id/train.tsv --dev ./data/fold_$fold_id/dev.tsv --test ./data/fold_$fold_id/test.tsv -o out_dir --num_epochs ${num_epochs} --batch_size 100 --prompt_id $prompt_id #> ./logs_self_attn/Bi_LSTM_Without_TRAITS/Prompt-$prompt_id/Fold-$fold_id.txt
		CUDA_VISIBLE_DEVICES=$gpu  python ATTN_STL.py --rnn_type 'Bi-LSTM' --word_feat_flag $word_feat_flag --sent_feat_flag $sent_feat_flag --essay_feat_flag $essay_feat_flag --ridley_feat_flag $ridley_feat_flag --word_feat_path $word_feat_path --sent_feat_path $sent_feat_path --essay_feat_path $essay_feat_path --ridley_feat_path $ridley_feat_path --prompt_filePath ../../attn-git/ASAP-Essay-Traits-Scoring/data/$prompt_id.words.txt --train ../../attn-git/ASAP-Essay-Traits-Scoring/data/CD-AEG/Prompt$prompt_id/fold_$fold_id/train.tsv --dev ../../attn-git/ASAP-Essay-Traits-Scoring/data/CD-AEG/Prompt$prompt_id/fold_$fold_id/dev.tsv --test ../../attn-git/ASAP-Essay-Traits-Scoring/data/CD-AEG/Prompt$prompt_id/fold_$fold_id/test.tsv -o out_dir --num_epochs ${num_epochs} --batch_size 100 --prompt_id $prompt_id --score_index $trait_id > ../logs_self_attn/STL/CD-AEG/Prompt-$prompt_id-Fold-$fold_id-Trait-$trait_id.txt
		done
		echo "Finished Trait: " $trait_id
	done
	echo "Finished fold id: " $fold_id
done
echo "Finished prompt id: " $prompt_id
