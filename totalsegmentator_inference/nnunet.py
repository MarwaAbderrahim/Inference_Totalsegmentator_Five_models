import os
import sys
import random
import string
import time
import shutil
import subprocess
from pathlib import Path
from os.path import join
import numpy as np
import nibabel as nib
from functools import partial
from p_tqdm import p_map
from multiprocessing import Pool
import tempfile
import torch
from totalsegmentator_inference.libs import nostdout

from totalsegmentator_inference.map_to_binary import class_map, class_map_5_parts, map_taskid_to_partname
from totalsegmentator_inference.alignment import as_closest_canonical_nifti, undo_canonical_nifti
from totalsegmentator_inference.alignment import as_closest_canonical, undo_canonical
from totalsegmentator_inference.resampling import change_spacing
from totalsegmentator_inference.libs import combine_masks, compress_nifti
from totalsegmentator_inference.dicom_io import dcm_to_nifti, save_mask_as_rtstruct
from totalsegmentator_inference.cropping import crop_to_mask_nifti, undo_crop_nifti
from totalsegmentator_inference.cropping import crop_to_mask, undo_crop
from totalsegmentator_inference.postprocessing import remove_outside_of_mask, extract_skin
from totalsegmentator_inference.nifti_ext_header import save_multilabel_nifti
import pathlib

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def _get_full_task_name(task_id: int, src: str="results"):
    src ="results"
    script_directory = pathlib.Path().resolve()
    path=os.path.dirname(__file__)
    modelpath=path + str("/models")
    base = Path(modelpath)
    dirs = [str(dir).split("/")[-1] for dir in base.glob("*")]
    # dirs contiendra les noms des fichiers et sous-dossiers dans le dossier "models"
    for dir in dirs:
        if f"Task{task_id:03d}" in dir:
            return dir
        
def contains_empty_img(imgs):
    """
    imgs: List of image pathes
    """
    is_empty = True
    for img in imgs:
        this_is_empty = len(np.unique(nib.load(img).get_fdata())) == 1
        is_empty = is_empty and this_is_empty
    return is_empty


def nnUNet_predict(dir_in, dir_out, task_id, model="3d_fullres", folds=None,
                   trainer="nnUNetTrainerV2", tta=False,
                   num_threads_preprocessing=6, num_threads_nifti_save=2):
    """
    Identical to bash function nnUNet_predict

    folds:  folds to use for prediction. Default is None which means that folds will be detected 
            automatically in the model output folder.
            for all folds: None
            for only fold 0: [0]
    """

    from nnunet.inference.predict import predict_from_folder
    from nnunet.paths import default_plans_identifier, network_training_output_dir, default_trainer

    save_npz = False
    lowres_segmentations = None
    part_id = 0
    num_parts = 1
    disable_tta = not tta
    overwrite_existing = False
    mode = "normal" if model == "2d" else "fastest"
    all_in_gpu = None
    step_size = 0.5
    chk = "model_final_checkpoint"
    disable_mixed_precision = False
    task_id = int(task_id)
    task_name = _get_full_task_name(task_id, src="results")
    print("task_name", task_name)
    plans_identifier = default_plans_identifier
    plan="nnUNetTrainerV2_ep4000_nomirror__nnUNetPlansv2.1"
    # model_folder_name = os.path.join('./model', task_name, plan)
    current_file_path = os.path.dirname(__file__)
    models='models'
    model_folder_name = os.path.abspath(os.path.join(current_file_path, models, task_name, plan))
    print("model_folder_name", model_folder_name)
    predict_from_folder(model_folder_name, dir_in, dir_out, folds, save_npz, num_threads_preprocessing,
                        num_threads_nifti_save, lowres_segmentations, part_id, num_parts, not disable_tta,
                        overwrite_existing=overwrite_existing, mode=mode, overwrite_all_in_gpu=all_in_gpu,
                        mixed_precision=not disable_mixed_precision,
                        step_size=step_size, checkpoint_name=chk)

def save_segmentation_nifti(class_map_item, tmp_dir=None, file_out=None, nora_tag=None, header=None, task_name=None, quiet=None):
    k, v = class_map_item
    if task_name != "total" and not quiet:
        print(f"Creating {v}.nii.gz")
    img = nib.load(tmp_dir / "s01.nii.gz")
    img_data = img.get_fdata()
    binary_img = img_data == k
    output_path = str(file_out / f"{v}.nii.gz")
    nib.save(nib.Nifti1Image(binary_img.astype(np.uint8), img.affine, header), output_path)
    if nora_tag != "None":
        subprocess.call(f"/opt/nora/src/node/nora -p {nora_tag} --add {output_path} --addtag mask", shell=True)


def nnUNet_predict_image(file_in, file_out, task_id, model="3d_fullres", folds=None,
                         trainer="nnUNetTrainerV2", tta=False, multilabel_image=True, 
                         resample=None, crop=None, crop_path=None, task_name="total", nora_tag="None", preview=False, 
                         save_binary=False, nr_threads_resampling=1, nr_threads_saving=6, force_split=False,
                         crop_addon=[3,3,3], roi_subset=None, output_type="nifti", 
                         statistics=False, quiet=False, verbose=False, test=0, skip_saving=False):
    """
    crop: string or a nibabel image
    resample: None or float  (target spacing for all dimensions)
    """
    file_in = Path(file_in)
    if file_out is not None:
        file_out = Path(file_out)
    if not file_in.exists():
        sys.exit("ERROR: The input file or directory does not exist.")
    multimodel = type(task_id) is list
    
    img_type = "nifti" if str(file_in).endswith(".nii") or str(file_in).endswith(".nii.gz") else "dicom"

    if img_type == "nifti" and output_type == "dicom":
        raise ValueError("To use output type dicom you also have to use a Dicom image as input.")
    
    tmp_dir = file_in.parent / ("nnunet_tmp_" + ''.join(random.Random().choices(string.ascii_uppercase + string.digits, k=8)))
    (tmp_dir).mkdir(exist_ok=True)
    with tmp_dir as tmp_folder:
    # with tempfile.TemporaryDirectory(prefix="nnunet_tmp_") as tmp_folder:
        tmp_dir = Path(tmp_folder)
        if img_type == "dicom":
            if not quiet: print("Converting dicom to nifti...")
            (tmp_dir / "dcm").mkdir()  # make subdir otherwise this file would be included by nnUNet_predict
            dcm_to_nifti(file_in, tmp_dir / "dcm" / "converted_dcm.nii.gz", verbose=verbose)
            file_in_dcm = file_in
            file_in = tmp_dir / "dcm" / "converted_dcm.nii.gz"
            if not quiet: print(f"  found image with shape {nib.load(file_in).shape}")
        

        
        # img_in_orig = nib.load(file_in).astype(np.float8)
        img_in_orig = nib.load(file_in)

        if len(img_in_orig.shape) == 2:
            raise ValueError("TotalSegmentator does not work for 2D images. Use a 3D image.")
        if len(img_in_orig.shape) > 3:
            print(f"WARNING: Input image has {len(img_in_orig.shape)} dimensions. Only using first three dimensions.")
            img_in_orig = nib.Nifti1Image(img_in_orig.get_fdata()[:,:,:,0], img_in_orig.affine)
        
        img_in = nib.Nifti1Image(img_in_orig.get_fdata().astype(np.float32), img_in_orig.affine)
        # copy img_in_orig
        print('task_id', task_id)
        print("crop", crop)
        print("crop_path", crop_path)
        # if crop is not None:
        # #     if crop == "lung" or crop == "pelvis" or crop == "heart":
        # #         combine_masks(crop_path, crop_path / f"{crop}.nii.gz", crop)
        #     if type(crop) is str:
        #        crop_mask_img = nib.load(crop_path / f"{crop}.nii.gz")
        #     else:
        #        crop_mask_img = crop
        #     img_in, bbox = crop_to_mask(img_in, crop_mask_img, addon=crop_addon, dtype=np.int32,
        #                                 verbose=verbose)
        #     if not quiet:
        #          print(f"  cropping from {crop_mask_img.shape} to {img_in.shape}")

        img_in = as_closest_canonical(img_in)

        if resample is not None:
            st = time.time()
            img_in_shape = img_in.shape
            img_in_zooms = img_in.header.get_zooms()
            img_in_rsp = change_spacing(img_in, [resample, resample, resample],
                                        order=3, dtype=np.int32, nr_cpus=nr_threads_resampling)  # 4 cpus instead of 1 makes it a bit slower
        else:
            img_in_rsp = img_in
        nib.save(img_in_rsp, tmp_dir / "s01_0000.nii.gz")

        # nr_voxels_thr = 512*512*900
        nr_voxels_thr = 256*256*900
        img_parts = ["s01"]
        ss = img_in_rsp.shape
        # If image to big then split into 3 parts along z axis. Also make sure that z-axis is at least 200px otherwise
        # splitting along it does not really make sense.
        do_triple_split = np.prod(ss) > nr_voxels_thr and ss[2] > 200 and multimodel
        if force_split:
            do_triple_split = True
        if do_triple_split:
            if not quiet: print(f"Splitting into subparts...")
            img_parts = ["s01", "s02", "s03"]
            third = img_in_rsp.shape[2] // 3
            margin = 20  # set margin with fixed values to avoid rounding problem if using percentage of third
            img_in_rsp_data = img_in_rsp.get_fdata()
            nib.save(nib.Nifti1Image(img_in_rsp_data[:, :, :third+margin], img_in_rsp.affine),
                    tmp_dir / "s01_0000.nii.gz")
            nib.save(nib.Nifti1Image(img_in_rsp_data[:, :, third+1-margin:third*2+margin], img_in_rsp.affine),
                    tmp_dir / "s02_0000.nii.gz")
            nib.save(nib.Nifti1Image(img_in_rsp_data[:, :, third*2+1-margin:], img_in_rsp.affine),
                    tmp_dir / "s03_0000.nii.gz")
            

        st = time.time()
        if multimodel:  # if running multiple models 
            # only compute model parts containing the roi subset
            if roi_subset is not None:
                part_names = []
                new_task_id = []
                for part_name, part_map in class_map_5_parts.items():
                    # print("part_name",part_name)
                    # print("part_map",part_map)
                    if any(organ in roi_subset for organ in part_map.values()):
                        # get taskid associated to model part_name
                        print("map_taskid_to_partname.items()", map_taskid_to_partname.items())
                        map_partname_to_taskid = {v:k for k,v in map_taskid_to_partname.items()}
                        new_task_id.append(map_partname_to_taskid[part_name])
                        part_names.append(part_name)
                task_id = new_task_id
                print("task_id",task_id)
                if verbose:
                    print(f"Computing parts: {part_names} based on the provided roi_subset")
            if test == 0:
                class_map_inv = {v: k for k, v in class_map[task_name].items()}
                (tmp_dir / "parts").mkdir(exist_ok=True)
                seg_combined = {}
                # iterate over subparts of image
                for img_part in img_parts:
                    img_shape = nib.load(tmp_dir / f"{img_part}_0000.nii.gz").shape
                    seg_combined[img_part] = np.zeros(img_shape, dtype=np.uint8)
                # Run several tasks and combine results into one segmentation
                for idx, tid in enumerate(task_id):
                    print(f"Predicting part {idx+1} of {len(task_id)} ...")
                    # with nostdout(verbose):
                    nnUNet_predict(tmp_dir, tmp_dir, tid, model, folds, trainer, tta,
                                       nr_threads_resampling, nr_threads_saving)
                    # iterate over models (different sets of classes)
                    for img_part in img_parts:
                        (tmp_dir / f"{img_part}.nii.gz").rename(tmp_dir / "parts" / f"{img_part}_{tid}.nii.gz")
                        seg = nib.load(tmp_dir / "parts" / f"{img_part}_{tid}.nii.gz").get_fdata()
                        for jdx, class_name in class_map_5_parts[map_taskid_to_partname[tid]].items():
                            seg_combined[img_part][seg == jdx] = class_map_inv[class_name]
                # iterate over subparts of image
                for img_part in img_parts:
                    nib.save(nib.Nifti1Image(seg_combined[img_part], img_in_rsp.affine), tmp_dir / f"{img_part}.nii.gz")
            elif test == 1:
                print("WARNING: Using reference seg instead of prediction for testing.")
                shutil.copy(Path("tests") / "reference_files" / "example_seg.nii.gz", tmp_dir / f"s01.nii.gz")
        else:
            if not quiet: print(f"Predicting...")
            if test == 0:
                # with nostdout(verbose):
                    nnUNet_predict(tmp_dir, tmp_dir, task_id, model, folds, trainer, tta,
                                   nr_threads_resampling, nr_threads_saving)
            # elif test == 2:
            #     print("WARNING: Using reference seg instead of prediction for testing.")
            #     shutil.copy(Path("tests") / "reference_files" / "example_seg_fast.nii.gz", tmp_dir / f"s01.nii.gz")
            elif test == 3:
                print("WARNING: Using reference seg instead of prediction for testing.")
                shutil.copy(Path("tests") / "reference_files" / "example_seg_lung_vessels.nii.gz", tmp_dir / f"s01.nii.gz")
        if not quiet: print("  Predicted in {:.2f}s".format(time.time() - st))

        # Combine image subparts back to one image
        if do_triple_split:
            combined_img = np.zeros(img_in_rsp.shape, dtype=np.uint8)
            combined_img[:,:,:third] = nib.load(tmp_dir / "s01.nii.gz").get_fdata()[:,:,:-margin]
            combined_img[:,:,third:third*2] = nib.load(tmp_dir / "s02.nii.gz").get_fdata()[:,:,margin-1:-margin]
            combined_img[:,:,third*2:] = nib.load(tmp_dir / "s03.nii.gz").get_fdata()[:,:,margin-1:]
            nib.save(nib.Nifti1Image(combined_img, img_in_rsp.affine), tmp_dir / "s01.nii.gz")

        img_pred = nib.load(tmp_dir / "s01.nii.gz")
        if preview:
            from totalsegmentator.preview import generate_preview
            # Generate preview before upsampling so it is faster and still in canonical space 
            # for better orientation.
            if not quiet: print("Generating preview...")
            st = time.time()
            smoothing = 20
            preview_dir = file_out.parent if multilabel_image else file_out
            generate_preview(img_in_rsp, preview_dir / f"preview_{task_name}.png", img_pred.get_fdata(), smoothing, task_name)
            if not quiet: print("  Generated in {:.2f}s".format(time.time() - st))

        if resample is not None:
            if not quiet: print("Resampling...")
            if verbose: print(f"  back to original shape: {img_in_shape}")    
            # Use force_affine otherwise output affine sometimes slightly off (which then is even increased
            # by undo_canonical)
            img_pred = change_spacing(img_pred, [resample, resample, resample], img_in_shape,
                                        order=0, dtype=np.uint8, nr_cpus=nr_threads_resampling, 
                                        force_affine=img_in.affine)

        if verbose: print("Undoing canonical...")
        img_pred = undo_canonical(img_pred, img_in_orig)

        # if crop is not None:
        #     if verbose: print("Undoing cropping...")
        #     img_pred = undo_crop(img_pred, img_in_orig, bbox)


        img_data = img_pred.get_fdata().astype(np.uint8)
        if save_binary:
            img_data = (img_data > 0).astype(np.uint8)

        if file_out is not None and skip_saving is False:
            if not quiet: print("Saving segmentations...")

            # Select subset of classes if required
            selected_classes = class_map[task_name]
            if roi_subset is not None:
                selected_classes = {k:v for k, v in selected_classes.items() if v in roi_subset}

            if output_type == "dicom":
                file_out.mkdir(exist_ok=True, parents=True)
                save_mask_as_rtstruct(img_data, selected_classes, file_in_dcm, file_out / "segmentations.dcm")
            else:
                new_header = img_in_orig.header.copy()
                new_header.set_data_dtype(np.uint8)

                st = time.time()
                if multilabel_image:
                    file_out.parent.mkdir(exist_ok=True, parents=True)
                else:
                    file_out.mkdir(exist_ok=True, parents=True)
                if multilabel_image:
                    img_out = nib.Nifti1Image(img_data, img_pred.affine, new_header)
                    save_multilabel_nifti(img_out, file_out, class_map[task_name])
                    if nora_tag != "None":
                        subprocess.call(f"/opt/nora/src/node/nora -p {nora_tag} --add {file_out} --addtag atlas", shell=True)
                else:  # save each class as a separate binary image
                    file_out.mkdir(exist_ok=True, parents=True)

                    if np.prod(img_data.shape) > 512*512*1000:
                        print(f"Shape of output image is very big. Setting nr_threads_saving=1 to save memory.")
                        nr_threads_saving = 1

                    # Code for single threaded execution  (runtime:24s)
                    if nr_threads_saving == 1:
                        for k, v in selected_classes.items():
                            binary_img = img_data == k
                            output_path = str(file_out / f"{v}.nii.gz")
                            nib.save(nib.Nifti1Image(binary_img.astype(np.uint8), img_pred.affine, new_header), output_path)
                            if nora_tag != "None":
                                subprocess.call(f"/opt/nora/src/node/nora -p {nora_tag} --add {output_path} --addtag mask", shell=True)
                    else:
                        nib.save(img_pred, tmp_dir / "s01.nii.gz")
                        _ = p_map(partial(save_segmentation_nifti, tmp_dir=tmp_dir, file_out=file_out, nora_tag=nora_tag, header=new_header, task_name=task_name, quiet=quiet),
                                selected_classes.items(), num_cpus=nr_threads_saving, disable=quiet)

            if not quiet: print(f"  Saved in {time.time() - st:.2f}s")

            # Postprocessing
            if task_name == "lung_vessels":
                remove_outside_of_mask(file_out / "lung_vessels.nii.gz", file_out / "lung.nii.gz")

            if task_name == "body":
                if not quiet: print("Creating body.nii.gz")
                combine_masks(file_out, file_out / "body.nii.gz", "body")
                if not quiet: print("Creating skin.nii.gz")
                skin = extract_skin(img_in_orig, nib.load(file_out / "body.nii.gz"))
                nib.save(skin, file_out / "skin.nii.gz")

    return nib.Nifti1Image(img_data, img_pred.affine)