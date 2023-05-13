for condition in "early_no_notes_no_waveforms" "early_no_waveforms" "early_with_ecg_quantiles" "early_only_ecg_quantiles"
do
    # Training
    ehrmeta=train_mixehr_metadata_${condition}.txt
    ehrdata=train_mixehr_${condition}.txt
    ./mixehr -f ./mixmimic/$ehrdata -m ./mixmimic/$ehrmeta -k 75 -i 500 --inferenceMethod JCVB0 --maxcores 8 --outputIntermediates

    # Get topics for patients
    for idset in "train" "vali" "test"
    do  
        ehrmeta=train_mixehr_metadata_${condition}.txt
        testdata=${idset}_mixehr_${condition}.txt
        trainedPrefix=./mixmimic/train_mixehr_${condition}_JCVB0_nmar_K75_iter500
        ./mixehr -m ./mixmimic/$ehrmeta -n JCVB0 --newPatsData ./mixmimic/$testdata --trainedModelPrefix $trainedPrefix -k 75 --inferNewPatientMetaphe --inferPatParams_maxiter 100
    done
done