#!/bin/bash

cd datasets

# Contact Natural Scenes for database access, create aws profile with your keys
export AWS_PROFILE="NaturalScenesDB"

# Define the subjects and directories
subjects=("subj01" "subj02" "subj03" "subj04" "subj05" "subj06" "subj07" "subj08")
bucket="s3://natural-scenes-dataset"
download_dir="datasets_untouched/NaturalScenes"
dataset_dir="snakemakePipeline/datasets/NaturalScenes"

Sync files for nsddata (anatomical)
for subject in "${subjects[@]}"; do
    echo "Syncing anatomical data for $subject from nsddata..."

    anat_dir="${download_dir}/nsddata/$subject/"

    # Sync files with specific names, excluding any with MNI in the filename
    aws s3 sync \
        "$bucket/nsddata/ppdata/$subject/anat/" $anat_dir \
        --exclude "*" \
        --include "HRT2*" \
        --include "aseg*" \
        --include "brainmask*" \
        --include "SWI*" \
        --include "T1*" \
        --include "T2*" \
        --include "TOF*" \
        --exclude "*MNI*"
done

# Copy and process files to match HCP format so ECP segmentation can be run
for subject in "${subjects[@]}"; do
    echo "Processing files for subject - $subject"

    source_subject_dir="$download_dir"
    source_data_dir="$source_subject_dir/nsddata/$subject"

    target_subject_dir="$dataset_dir/$subject"
    target_structural_dir="${target_subject_dir}/structural"
    target_SWI_dir="${target_subject_dir}/SWI"

    mkdir -p $target_structural_dir
    mkdir -p $target_SWI_dir

    # Resample T1, T2, and brainmask to match HCP structural resolution using mrgrid
    echo "Resampling T1, T2, and brainmask to match HCP T1 resolution..."
    
    t1w_brain="$target_structural_dir/T1w_brain.nii.gz"
    t2w_brain="$target_structural_dir/T2w_brain.nii.gz"
    
    HCP_T1_TEMPLATE="Pipeline/datasets/HCP_1200/130114/structural/T1w_brain.nii.gz"
    HCP_T2_TEMPLATE="Pipeline/datasets/HCP_1200/130114/structural/T2w_brain.nii.gz"

    mrgrid "$source_data_dir/T1_0pt8_masked.nii.gz" regrid -template "$HCP_T1_TEMPLATE" $t1w_brain -force
    mrgrid "$source_data_dir/T2_0pt8_masked.nii.gz" regrid -template "$HCP_T1_TEMPLATE" $t2w_brain -force
    mrgrid "$source_data_dir/brainmask_0pt8.nii.gz" regrid -template "$HCP_T1_TEMPLATE" "$target_structural_dir/brainmask.nii.gz" -force
    
    # Perform histogram matching to rescale the T1w and T2w images
    echo "Performing histogram matching for T1 and T2..."
    mrhistmatch scale "$t1w_brain" "$HCP_T1_TEMPLATE" "$t1w_brain" -force
    mrhistmatch scale "$t2w_brain" "$HCP_T2_TEMPLATE" "$t2w_brain" -force

    # Create symbolic links into the formatted dataset folder
    cp -l -f "$target_structural_dir/T1w_brain.nii.gz" "$target_structural_dir/T1w.nii.gz"
    cp -l -f "$target_structural_dir/T2w_brain.nii.gz" "$target_structural_dir/T2w.nii.gz"    

done


