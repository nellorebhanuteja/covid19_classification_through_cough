stage=

listsdir=
audiodir=

metadatafile=

datadir=
feature_dir=
output_dir=
mkdir -p $output_dir

train_config=
feats_config=
. parse_options.sh


if [ $stage -le 0 ]; then
    echo "==== Preparing data folders ====="
    mkdir -p $datadir
    cat $listsdir/*.txt | sort | uniq >$datadir/allfiles.scp
    awk -v audiodir=$audiodir {'print $1" "audiodir"/"$1".flac"'} < $datadir/allfiles.scp >$datadir/wav.scp
    cat $metadatafile | awk {'split($1,a,",");print a[1]" "a[2]'} >$datadir/labels
    for fold in $(seq 1 5);do
        mkdir -p $datadir/fold_$fold
        for item in train val;do
            awk -v audiodir=$audiodir {'print $1" "audiodir"/"$1".flac"'} < $listsdir/${item}_fold_$fold.txt >$datadir/fold_$fold/$item.scp 
            awk 'NR==FNR{_[$1];next}($1 in _)' $listsdir/${item}_fold_${fold}.txt $datadir/labels >$datadir/fold_$fold/${item}_labels
        done
    done
fi

if [ $stage -le 1 ]; then
    # Creates a separate pickle file containing feature matrix for each recording in the wav.scp
    # Expects a data folder, with train_dev and eval folders inside. each folder has a wav.scp file
    # Each row in wav.scp is formatted as: <wav_id> <wav_file_path>
    # Feature matrices are written to: $feature_dir/{train_dev/eval_set}/<wav_id>_<feature_type>.pkl
    # feature.conf specifies configuration settings for feature extraction
    echo "==== Feature extraction ====="
    mkdir -p $feature_dir
    python feature_extraction.py -c $feats_config -i $datadir/wav.scp -o $feature_dir
    cp $feature_dir/feats.scp $datadir/feats.scp
fi
# Logistic Regression
if [ $stage -le 2 ]; then
    output_dir_lr="${output_dir}/results_lr"
    train_config='conf/train_lr.conf'
    mkdir -p $output_dir_lr
    echo "========= Logistic regression classifier ======================"
    cat $train_config
    for fold in $(seq 1 5);do
        mkdir -p $output_dir_lr/fold_${fold}
        cp $datadir/feats.scp $datadir/fold_${fold}/
        # Train
        python train.py -c $train_config -d $datadir/fold_${fold} -o $output_dir_lr/fold_${fold}
        # Validate
        python infer.py --modelfil $output_dir_lr/fold_${fold}/model.pkl --featsfil $datadir/fold_${fold}/feats.scp --file_list $datadir/fold_${fold}/val.scp --outfil $output_dir_lr/fold_${fold}/val_scores.txt -c $train_config
        # Score
        python scoring.py --ref_file $datadir/fold_${fold}/val_labels --target_file $output_dir_lr/fold_${fold}/val_scores.txt --output_file $output_dir_lr/fold_${fold}/val_results.pkl
    done
    # below file can be uploaded to scoring server to appear on leaderboard for development set performance 
    cat $output_dir_lr/fold_1/val_scores.txt $output_dir_lr/fold_2/val_scores.txt $output_dir_lr/fold_3/val_scores.txt $output_dir_lr/fold_4/val_scores.txt $output_dir_lr/fold_5/val_scores.txt > $output_dir_lr/val_scores_allfolds.txt
    # summarize all folds performance
    python summarize.py $output_dir_lr
fi
# Random Forest 
if [ $stage -le 3 ]; then
    output_dir_rf="${output_dir}/results_rf"
    train_config='conf/train_rf.conf'
    mkdir -p $output_dir_rf
    echo "========= Random forest classifier ======================"
    cat $train_config
    for fold in $(seq 1 5);do
        mkdir -p $output_dir_rf/fold_${fold}
        cp $datadir/feats.scp $datadir/fold_${fold}/
        # Train
        python train.py -c $train_config -d $datadir/fold_${fold} -o $output_dir_rf/fold_${fold}
        # Validate
        python infer.py --modelfil $output_dir_rf/fold_${fold}/model.pkl --featsfil $datadir/fold_${fold}/feats.scp --file_list $datadir/fold_${fold}/val.scp --outfil $output_dir_rf/fold_${fold}/val_scores.txt -c $train_config
        # Score
        python scoring.py --ref_file $datadir/fold_${fold}/val_labels --target_file $output_dir_rf/fold_${fold}/val_scores.txt --output_file $output_dir_rf/fold_${fold}/val_results.pkl 
    done
    # below file can be uploaded to scoring server to appear on leaderboard for development set performance 
    cat $output_dir_rf/fold_1/val_scores.txt $output_dir_rf/fold_2/val_scores.txt $output_dir_rf/fold_3/val_scores.txt $output_dir_rf/fold_4/val_scores.txt $output_dir_rf/fold_5/val_scores.txt > $output_dir_rf/val_scores_allfolds.txt
    # summarize all folds performance
    python summarize.py $output_dir_rf
fi
# Multi-Layer Perceptron
if [ $stage -le 4 ]; then
    output_dir_mlp="${output_dir}/results_mlp"
    train_config='conf/train_mlp.conf'
    mkdir -p $output_dir_mlp
    echo "========= Multilayer perceptron classifier ======================"
    cat $train_config
    for fold in $(seq 1 5);do
        mkdir -p $output_dir_mlp/fold_${fold}
        cp $datadir/feats.scp $datadir/fold_${fold}/
        # Train
        python train.py -c $train_config -d $datadir/fold_${fold} -o $output_dir_mlp/fold_${fold}
        # Validate
        python infer.py --modelfil $output_dir_mlp/fold_${fold}/model.pkl --featsfil $datadir/fold_${fold}/feats.scp --file_list $datadir/fold_${fold}/val.scp --outfil $output_dir_mlp/fold_${fold}/val_scores.txt -c $train_config
        # Score
        python scoring.py --ref_file $datadir/fold_${fold}/val_labels --target_file $output_dir_mlp/fold_${fold}/val_scores.txt --output_file $output_dir_mlp/fold_${fold}/val_results.pkl 
    done
    # below file can be uploaded to scoring server to appear on leaderboard for development set performance 
    cat $output_dir_mlp/fold_1/val_scores.txt $output_dir_mlp/fold_2/val_scores.txt $output_dir_mlp/fold_3/val_scores.txt $output_dir_mlp/fold_4/val_scores.txt $output_dir_mlp/fold_5/val_scores.txt > $output_dir_mlp/val_scores_allfolds.txt
    # summarize all folds performance
    python summarize.py $output_dir_mlp
fi
# SVM 
if [ $stage -le 5 ]; then
    output_dir_rf="${output_dir}/results_svm"
    train_config='conf/train_svm.conf'
    mkdir -p $output_dir_rf
    echo "========= SVM classifier ======================"
    cat $train_config
    for fold in $(seq 1 5);do
        mkdir -p $output_dir_rf/fold_${fold}
        cp $datadir/feats.scp $datadir/fold_${fold}/
        # Train
        python train.py -c $train_config -d $datadir/fold_${fold} -o $output_dir_rf/fold_${fold}
        # Validate
        python infer.py --modelfil $output_dir_rf/fold_${fold}/model.pkl --featsfil $datadir/fold_${fold}/feats.scp --file_list $datadir/fold_${fold}/val.scp --outfil $output_dir_rf/fold_${fold}/val_scores.txt -c $train_config
        # Score
        python scoring.py --ref_file $datadir/fold_${fold}/val_labels --target_file $output_dir_rf/fold_${fold}/val_scores.txt --output_file $output_dir_rf/fold_${fold}/val_results.pkl 
    done
    # below file can be uploaded to scoring server to appear on leaderboard for development set performance 
    cat $output_dir_rf/fold_1/val_scores.txt $output_dir_rf/fold_2/val_scores.txt $output_dir_rf/fold_3/val_scores.txt $output_dir_rf/fold_4/val_scores.txt $output_dir_rf/fold_5/val_scores.txt > $output_dir_rf/val_scores_allfolds.txt
    # summarize all folds performance
    python summarize.py $output_dir_rf
fi
echo "Done!!!"