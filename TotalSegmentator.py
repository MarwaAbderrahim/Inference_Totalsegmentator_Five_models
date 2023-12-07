#!/usr/bin/env python
import os
import io
import sys
import warnings
sys.stdout =io.StringIO()
import time
import nibabel
import argparse
import numpy as np
import nibabel as nib 
from pathlib import Path
import SimpleITK as sitk
from pkg_resources import require
from contextlib import contextmanager
from DicomRTTool import DicomReaderWriter
from totalsegmentator_inference.python_api import totalsegmentator
from totalsegmentator_inference.libs import combine_masks_to_multilabel_file

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def main():

    output = io.StringIO()
    sys.stdout = output

    parser = argparse.ArgumentParser(description="Segment 104 anatomical structures in CT images.")

    parser.add_argument("--test", metavar="0|1|2|3", choices=[0, 1, 2, 3], type=int,
                         help="Only needed for unittesting.",
                         default=0)
    
    parser.add_argument("-mlabels", "--mlabels", action="store_true", help="Save one multilabel image for all classes",
                          default=True)
    
    parser.add_argument("-i", metavar="filepath", dest="input",
                         help="CT nifti image or folder of dicom slices", 
                         type=lambda p: Path(p).absolute(), required=True)

    parser.add_argument("-o", metavar="directory", dest="output",
                         help="Output directory for segmentation masks", 
                         type=lambda p: Path(p).absolute(), required=True)

    parser.add_argument("-ot", "--output_type", choices=["nifti", "dicom"],
                    help="Select if segmentations shall be saved as Nifti or as Dicom RT Struct image.",
                    default="nifti")
                    
    parser.add_argument("-ml", "--ml", action="store_true", help="Save one multilabel image for all classes",
                        default=False)

    parser.add_argument("-nr", "--nr_thr_resamp", type=int, help="Nr of threads for resampling", default=1)

    parser.add_argument("-ns", "--nr_thr_saving", type=int, help="Nr of threads for saving segmentations", 
                        default=6)

    parser.add_argument("-f", "--fast", action="store_true", help="Run faster lower resolution model",
                        default=False)
    
    parser.add_argument("-om", "--onemodel", action="store_true", help="Run faster lower resolution model",
                        default=False)

    parser.add_argument("-t", "--nora_tag", type=str, 
                        help="tag in nora as mask. Pass nora project id as argument.",
                        default="None")

    parser.add_argument("-p", "--preview", action="store_true", 
                        help="Generate a png preview of segmentation",
                        default=False)

    parser.add_argument("-ta", "--task", choices=["total", "lung_vessels", "cerebral_bleed", 
                        "hip_implant", "coronary_arteries", "body", "pleural_pericard_effusion", 
                        "liver_vessels", "bones_extremities", "tissue_types",
                        "heartchambers_highres", "head", "aortic_branches", "heartchambers_test", 
                        "bones_tissue_test", "aortic_branches_test", "test"],
                        help="Select which model to use. This determines what is predicted.",
                        default="total")

    parser.add_argument("-rs", "--roi_subset", type=str, nargs="+",
                        help="Define a subset of classes to save (space separated list of class names). If running 1.5mm model, will only run the appropriate models for these rois.")

    parser.add_argument("-s", "--statistics", action="store_true", 
                        help="Calc volume (in mm3) and mean intensity. Results will be in statistics.json",
                        default=False)

    parser.add_argument("-r", "--radiomics", action="store_true", 
                        help="Calc radiomics features. Requires pyradiomics. Results will be in statistics_radiomics.json",
                        default=False)

    parser.add_argument("-cp", "--crop_path", help="Custom path to masks used for cropping. If not set will use output directory.", 
                        type=lambda p: Path(p).absolute(), default=None)

    parser.add_argument("-bs", "--body_seg", action="store_true", 
                        help="Do initial rough body segmentation and crop image to body region",
                        default=False)
    
    parser.add_argument("-fs", "--force_split", action="store_true", help="Process image in 3 chunks for less memory consumption",
                        default=True)

    parser.add_argument("-ss", "--skip_saving", action="store_true", 
                        help="Skip saving of segmentations for faster runtime if you are only interested in statistics.",
                        default=False)

    parser.add_argument("-q", "--quiet", action="store_true", help="Print no intermediate outputs",
                        default=False)

    parser.add_argument("-v", "--verbose", action="store_true", help="Show more intermediate output",
                        default=False)

    args = parser.parse_args()

    #-------------------------------------- Convert dicom to nifti----------------------------------------------

    output_dcm=args.output

    sys.stdout = sys.__stdout__
    #-------------------------------------- Générer le masque ----------------------------------------------
    
    totalsegmentator(args.input, args.output, args.ml, args.nr_thr_resamp, args.nr_thr_saving,
                     args.fast, args.nora_tag, args.preview, args.task, args.roi_subset,
                     args.statistics, args.radiomics, args.crop_path, args.body_seg,
                     args.force_split, args.output_type, args.quiet, args.verbose, args.test, args.onemodel)

    #-------------------------------------- Combine mask ----------------------------------------------
    
    sys.stdout = sys.__stdout__

    sys.stdout = io.StringIO()
    multilabel_file = str(args.output) + "/multilabel.nii.gz"
    
    if args.mlabels : 
          combine_masks_to_multilabel_file(args.output, multilabel_file)

if __name__ == "__main__":
    main()

