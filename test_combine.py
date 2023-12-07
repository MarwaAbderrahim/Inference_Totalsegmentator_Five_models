import io
import os
import contextlib
import sys
import time
import shutil
import zipfile
from pathlib import Path
import argparse

import requests
import numpy as np
import nibabel as nib

from totalsegmentator_inference.map_to_binary import class_map
from totalsegmentator_inference.map_to_binary import class_map_5_parts




parser = argparse.ArgumentParser(description='Register images from input images and reference image')
parser.add_argument("-i", "--input", type=str, required=True, help="path to input images")    
# parser.add_argument("-i", "--input", type=Path, default=Path(__file__).absolute().parent / "data", help="Path to the data directory",) 
parser.add_argument("-o", "--output", type=str, help="path to output registred images")                  #output segmented mask we can call by "-o" command

parser.add_argument("-roi", "--roi", help="Save one roi", choices=["ribs", "muscles", "cardiac", "vertebrae", "vertebrae_ribs",
                            "lung", "lung_right","lung_left","Abdomen", "heart", "pelvis", "pelvisVer"], default=False)


def combine_masks_to_multilabel_file(masks_dir, multilabel_file, class_type):
    """
    Generate one multilabel nifti file from a directory of single binary masks of each class.
    This multilabel file is needed to train a nnU-Net.

    masks_dir: path to directory containing all the masks for one subject
    multilabel_file: path of the output file (a nifti file)
    """
    if class_type == "ribs":
        masks = list(class_map_5_parts["class_map_part_ribs"].values())
    elif class_type=="muscles":
        masks=list(class_map_5_parts["class_map_part_muscles"].values())
    elif class_type == "vertebrae":
        masks = list(class_map_5_parts["class_map_part_vertebrae"].values())
    elif class_type =="cardiac":
        masks= list(class_map_5_parts["class_map_part_cardiac"].values())
    elif class_type == "vertebrae_ribs":
        masks = list(class_map_5_parts["class_map_part_vertebrae"].values()) + list(class_map_5_parts["class_map_part_ribs"].values())
    elif class_type == "lung":
        masks = ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
                 "lung_middle_lobe_right", "lung_lower_lobe_right"]
    elif class_type == "lung_left":
        masks = ["lung_upper_lobe_left", "lung_lower_lobe_left"]
    elif class_type == "lung_right":
        masks = ["lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"]
    elif class_type == "heart":
        masks = ["heart_myocardium", "heart_atrium_left", "heart_ventricle_left",
                 "heart_atrium_right", "heart_ventricle_right"]
    elif class_type == "pelvis":
        # masks = ["hip_left", "hip_right", "sacrum", "vertebrae_L4", "vertebrae_L5", "femur_left", "femur_right"]
        masks = ["hip_left", "hip_right", "sacrum", "vertebrae_L5", "vertebrae_L4"]

    elif class_type == "pelvisVer":
        # masks = ["hip_left", "hip_right", "sacrum", "vertebrae_L4", "vertebrae_L5", "femur_left", "femur_right"]
        masks = ["hip_left", "hip_right", "sacrum", "vertebrae_L5", "vertebrae_L4"] + list(class_map_5_parts["class_map_part_vertebrae"].values())

    elif class_type == "body":
        masks = ["body_trunc", "body_extremities"]
   
    elif class_type == "Abdomen":
        masks = ["spleen", "kidney_right", "kidney_left", "gallbladder", "esophagus", "liver", "stomach", "aorta", "inferior_vena_cava",
                  "portal_vein_and_splenic_vein", "pancreas", "adrenal_gland_right", "adrenal_gland_left"]
    elif class_type == "Total":
        masks = list(class_map_5_parts["class_map_part_organs"].values()) + list(class_map_5_parts["class_map_part_ribs"].values()) 
        + list(class_map_5_parts["class_map_part_vertebrae"].values())  + list(class_map_5_parts["class_map_part_cardiac"].values())
        + list(class_map_5_parts["class_map_part_muscles"].values())
    
    masks_dir = Path(masks_dir)
    ref_img = nib.load(masks_dir / "liver.nii.gz")
    # masks = class_map["total"].values()
    img_out = np.zeros(ref_img.shape).astype(np.uint8)

    for idx, mask in enumerate(masks):
        if os.path.exists(f"{masks_dir}/{mask}.nii.gz"):
            img = nib.load(f"{masks_dir}/{mask}.nii.gz").get_fdata()
        else:
            print(f"Mask {mask} is missing. Filling with zeros.")
            img = np.zeros(ref_img.shape)
        img_out[img > 0.5] = idx+1

    mask_reoriented=nib.Nifti1Image(img_out, affine=ref_img.affine)

    # target_orientation = nib.orientations.io_orientation(ref_img.affine)
    # mask_reoriented = nib.as_closest_canonical(mask_reoriented)

    # current_orientation =  nib.orientations.io_orientation(mask_reoriented.affine)

    # if not np.array_equal(current_orientation, target_orientation):
    #     inv_ornt, inv_affine = nib.orientations.apply_orientation(mask_reoriented.affine, mask_reoriented.shape)
    #     mask_reoriented = nib.Nifti1Image(np.flip(mask_reoriented.get_fdata(), axis=0), affine=inv_affine)
    nib.save(mask_reoriented, multilabel_file)



args = parser.parse_args()
masks_dir=args.input
out_path=args.output
roi=args.roi
combine_masks_to_multilabel_file(masks_dir, out_path, roi)
