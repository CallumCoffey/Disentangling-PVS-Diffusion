#!/bin/bash

cd datasets

# Create a connectomeDB profile, get access keys from the HCP and use that profile here
export AWS_PROFILE="connectomeDB"

# Define the subjects and directories
group=("HCP_1200")
subjects=("100610" "130114" "165436" "167440" "239136")

# subjects=(
#     "102109" "102614" "102715" "106824" "111211" "113316" "115724" "117021"
#     "118831" "120414" "123723" "125222" "125424" "126426" "127226" "130114"
#     "130720" "135124" "135629" "138332" "139435" "143224" "146735" "147636"
#     "151324" "151930" "152225" "152427" "153126" "161832" "165436" "167440" "168947" "169545"
#     "175136" "176845" "180230" "186545" "186848" "188145" "191235" "192237" "194443"
#     "198047" "199352" "200513" "206323" "206525" "206727" "206828" "210112" "211619" "211821"
#     "213017" "213522" "219231" "255740" "274542" "281135" "300719" "325129" "329844"
#     "349244" "350330" "368753" "376247" "378756" "392447" "394956" "419239" "421226"
#     "453542" "454140" "463040" "510225" "516742" "519647" "541640" "552241"
#     "558960" "589567" "634748" "635245" "654552" "675661" "680452" "692964"
#     "694362" "728454" "757764" "774663" "788674" "814548" "825654" "828862" "832651"
#     "869472" "878877" "886674" "905147" "908860" "911849" "933253" "971160" "987074" "989987"
# )

bucket="s3://hcp-openaccess"
download_dir="datasets_untouched/$group"
dataset_dir="Pipeline/datasets/$group"
    
mkdir -p $dataset_dir

# Sync files for nsddata (anatomical)
for subject in "${subjects[@]}"; do
    echo "Syncing data for $subject from $group..."

    # Sync files
    aws s3 sync \
        "$bucket/$group/$subject/T1w/" "$download_dir/$subject/" \
        --exclude "*" \
        --include "T*w_acpc_dc_restore.nii.gz" \
        --include "T*w_acpc_dc_restore_brain.nii.gz" \
        --include "brainmask_fs.nii.gz*" \
        --include "Diffusion_7T/*"
done

# Copy and process files to match HCP format so ECP segmentation can be run
for subject in "${subjects[@]}"; do
    echo "Processing files for subject - $subject"

    source_subject_dir="$download_dir/$subject"
    source_structural_dir="$source_subject_dir"
    source_diffusion_dir="$source_subject_dir/Diffusion_7T"

    target_subject_dir="$dataset_dir/$subject"
    target_structural_dir="$target_subject_dir/structural"
    target_diffusion_dir="$target_subject_dir/diffusion"

    mkdir -p $target_structural_dir
    mkdir -p $target_diffusion_dir
   
    # Create symbolic links into the formatted dataset folder
    cp -l -f "$source_structural_dir/T1w_acpc_dc_restore.nii.gz" "$target_structural_dir/T1w.nii.gz"
    cp -l -f "$source_structural_dir/T2w_acpc_dc_restore.nii.gz" "$target_structural_dir/T2w.nii.gz"    
    cp -l -f "$source_structural_dir/T1w_acpc_dc_restore_brain.nii.gz" "$target_structural_dir/T1w_brain.nii.gz"
    cp -l -f "$source_structural_dir/T2w_acpc_dc_restore_brain.nii.gz" "$target_structural_dir/T2w_brain.nii.gz"
    cp -l -f "$source_structural_dir/brainmask_fs.nii.gz" "$target_structural_dir/brainmask.nii.gz"

    cp -l -f "$source_diffusion_dir/data.nii.gz" "$target_diffusion_dir/DWI.nii.gz"
    cp -l -f "$source_diffusion_dir/bvals" "$target_diffusion_dir/bvals"
    cp -l -f "$source_diffusion_dir/bvecs" "$target_diffusion_dir/bvecs"
    cp -l -f "$source_diffusion_dir/nodif_brain_mask.nii.gz" "$target_diffusion_dir/diffusion_mask.nii.gz"

done
