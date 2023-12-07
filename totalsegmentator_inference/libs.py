import io
import os
import contextlib
import sys
import time
import shutil
import zipfile
from pathlib import Path

import requests
import numpy as np
import nibabel as nib

from totalsegmentator_inference.map_to_binary import class_map
from totalsegmentator_inference.map_to_binary import class_map_5_parts


class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostdout(verbose=False):
    if not verbose:
        save_stdout = sys.stdout
        sys.stdout = DummyFile()
        yield
        sys.stdout = save_stdout
    else:
        yield

def get_config_dir():
    base_dir = Path("./models/nnUNet")
    config_dir = base_dir.parent
    print("config_dir", config_dir)
    return config_dir


def setup_nnunet():
    config_dir="./models/"
    weights_dir = config_dir / "nnunet/"

    # This variables will only be active during the python script execution. Therefore
    # we do not have to unset them in the end.
    os.environ["nnUNet_raw_data_base"] = str(weights_dir)  # not needed, just needs to be an existing directory
    os.environ["nnUNet_preprocessed"] = str(weights_dir)  # not needed, just needs to be an existing directory
    os.environ["RESULTS_FOLDER"] = str(weights_dir)


def combine_masks_to_multilabel_file(masks_dir, multilabel_file):
    """
    Generate one multilabel nifti file from a directory of single binary masks of each class.
    This multilabel file is needed to train a nnU-Net.

    masks_dir: path to directory containing all the masks for one subject
    multilabel_file: path of the output file (a nifti file)
    """
    masks_dir = Path(masks_dir)
    ref_img = nib.load(masks_dir / "liver.nii.gz")
    masks = class_map["total"].values()
    img_out = np.zeros(ref_img.shape).astype(np.uint8)

    for idx, mask in enumerate(masks):
        if os.path.exists(f"{masks_dir}/{mask}.nii.gz"):
            img = nib.load(f"{masks_dir}/{mask}.nii.gz").get_fdata()
        else:
            print(f"Mask {mask} is missing. Filling with zeros.")
            img = np.zeros(ref_img.shape)
        img_out[img > 0.5] = idx + 1

    mask_reoriented = nib.Nifti1Image(img_out, ref_img.affine)
    type(mask_reoriented)
    target_orientation = nib.orientations.io_orientation(ref_img.affine)
    # mask_reoriented = nib.as_closest_canonical(mask_reoriented)
    current_orientation = nib.orientations.io_orientation(mask_reoriented.affine)
    mask_reoriented = nib.Nifti1Image(img_out, ref_img.affine)
    target_orientation = nib.orientations.io_orientation(ref_img.affine)
    current_orientation = nib.orientations.io_orientation(mask_reoriented.affine)
    type(target_orientation)
    type(current_orientation)

    if not np.array_equal(np.asarray(current_orientation), np.asarray(target_orientation)):
        mask_reoriented = nib.as_closest_canonical(mask_reoriented)

    nib.save(mask_reoriented, multilabel_file)

def combine_masks(mask_dir, output, class_type):
    """
    Combine classes to masks

    mask_dir: directory of totalsegmetator masks
    output: output path
    class_type: ribs | vertebrae | vertebrae_ribs | lung | heart
    """
    if class_type == "ribs":
        masks = list(class_map_5_parts["class_map_part_ribs"].values())
    elif class_type == "vertebrae":
        masks = list(class_map_5_parts["class_map_part_vertebrae"].values())
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
        masks = ["femur_left", "femur_right", "hip_left", "hip_right", "sacrum"]
    elif class_type == "body":
        masks = ["body_trunc", "body_extremities"]

    ref_img = None
    for mask in masks:
        if (mask_dir / f"{mask}.nii.gz").exists():
            ref_img = nib.load(mask_dir / f"{masks[0]}.nii.gz")
        else:
            raise ValueError(f"Could not find {mask_dir / mask}.nii.gz. Did you run TotalSegmentator successfully?")

    combined = np.zeros(ref_img.shape, dtype=np.uint8)
    for idx, mask in enumerate(masks):
        if (mask_dir / f"{mask}.nii.gz").exists():
            img = nib.load(mask_dir / f"{mask}.nii.gz").get_fdata()
            combined[img > 0.5] = 1

    nib.save(nib.Nifti1Image(combined, ref_img.affine), output)


def compress_nifti(file_in, file_out, dtype=np.int32, force_3d=True):
    img = nib.load(file_in)
    data = img.get_fdata()
    if force_3d and len(data.shape) > 3:
        print("Info: Input image contains more than 3 dimensions. Only keeping first 3 dimensions.")
        data = data[:,:,:,0]
    new_image = nib.Nifti1Image(data.astype(dtype), img.affine)
    nib.save(new_image, file_out)

def combine_roi_to_multilabel_file(masks_dir, multilabel_file, class_type):
    """
    Generate one multilabel nifti file from a directory of single binary masks of each class.
    This multilabel file is needed to train a nnU-Net.

    masks_dir: path to directory containing all the masks for one subject
    multilabel_file: path of the output file (a nifti file)
    """

    if class_type == "ribs":
        masks = list(class_map_5_parts["class_map_part_ribs"].values())
    elif class_type == "vertebrae":
        masks = list(class_map_5_parts["class_map_part_vertebrae"].values())
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
        masks = ["hip_left", "hip_right", "sacrum", "vertebrae_L4", "vertebrae_L5", "femur_left", "femur_right"]
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

    nib.save(nib.Nifti1Image(img_out, None), multilabel_file)

def check_if_shape_and_affine_identical(img_1, img_2):
    
    if not np.array_equal(img_1.affine, img_2.affine):
        print("Affine in:")
        print(img_1.affine)
        print("Affine out:")
        print(img_2.affine)
        print("Diff:")
        print(np.abs(img_1.affine-img_2.affine))
        print("WARNING: Output affine not equal to input affine. This should not happen.")

    if img_1.shape != img_2.shape:
        print("Shape in:")
        print(img_1.shape)
        print("Shape out:")
        print(img_2.shape)
        print("WARNING: Output shape not equal to input shape. This should not happen.")