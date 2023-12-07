# TotalSegmentator

Tool for segmentation of 104 classes in CT images. It was trained on a wide range of different CT images (different scanners, institutions, protocols,...) and therefore should work well on most images. 

### Installation

The inference code works on Windows and on CPU and GPU (on CPU it is slow).

Install dependencies:  
* Python >= 3.7
* [Pytorch](http://pytorch.org/)
* You should not have any nnU-Net installation in your python environment since TotalSegmentator will install its own custom installation.
* PS: To test the segmentation code, you need to have a GPU available with at least 12GB of memory per user.

### Install Totalsegmentator
```
Go to the installation folder and follow the steps

```

### One last step :
To complete the setup process, please follow these instructions:

Download the trained model using this link (I couldn't upload it due to its size exceeding 25MB) : https://abysmedical-my.sharepoint.com/:u:/g/personal/mabderrahim_abys-medical_com/ESmywf-ymBRIlsbIzu5SPf8BoFaxvmuEwyd8VkFItuXW6A?e=9Z7UWj

Once downloaded, place the .model file in the following directory:

/totalsegmentator_inference/models/Task257_TotalSegmentator_15mm_1139subj/nnUNetTrainerV2_ep4000_nomirror__nnUNetPlansv2.1/fold_0



### Usage
In orfer to test the code, you need to download all files in this git.
```
python TotalSegmentator.py -i "path of nifit image" -o "path of the output folder"
```

### Other commands
If you want to combine some subclasses (e.g. pelvic) into one multilabel mask you can use the following command:
```
python test_combine.py -i "totalsegmentator_output_dir" -o "combined_mask.nii.gz" -roi "pelvis"
```

### Typical problems
When you get the following error message
```
ITK ERROR: ITK only supports orthonormal direction cosines. No orthonormal definition found!
```
you should do
```
pip install SimpleITK==2.0.2
```

### Class details

The following table shows a list of all classes.

TA2 is a standardised way to name anatomy. Mostly the TotalSegmentator names follow this standard. 
For some classes they differ which you can see in the table below.

|TotalSegmentator name|TA2 name|
|:-----|:-----|
spleen ||
kidney_right ||
kidney_left ||
gallbladder ||
liver ||
stomach ||
aorta ||
inferior_vena_cava ||
portal_vein_and_splenic_vein | hepatic portal vein |
pancreas ||
adrenal_gland_right | suprarenal gland |
adrenal_gland_left | suprarenal gland |
lung_upper_lobe_left | superior lobe of left lung |
lung_lower_lobe_left | inferior lobe of left lung |
lung_upper_lobe_right | superior lobe of right lung |
lung_middle_lobe_right | middle lobe of right lung |
lung_lower_lobe_right | inferior lobe of right lung |
vertebrae_L5 ||
vertebrae_L4 ||
vertebrae_L3 ||
vertebrae_L2 ||
vertebrae_L1 ||
vertebrae_T12 ||
vertebrae_T11 ||
vertebrae_T10 ||
vertebrae_T9 ||
vertebrae_T8 ||
vertebrae_T7 ||
vertebrae_T6 ||
vertebrae_T5 ||
vertebrae_T4 ||
vertebrae_T3 ||
vertebrae_T2 ||
vertebrae_T1 ||
vertebrae_C7 ||
vertebrae_C6 ||
vertebrae_C5 ||
vertebrae_C4 ||
vertebrae_C3 ||
vertebrae_C2 ||
vertebrae_C1 ||
esophagus ||
trachea ||
heart_myocardium ||
heart_atrium_left ||
heart_ventricle_left ||
heart_atrium_right ||
heart_ventricle_right ||
pulmonary_artery | pulmonary arteries |
brain ||
iliac_artery_left | common iliac artery |
iliac_artery_right | common iliac artery |
iliac_vena_left | common iliac vein |
iliac_vena_right | common iliac vein |
small_bowel | small intestine |
duodenum ||
colon ||
rib_left_1 ||
rib_left_2 ||
rib_left_3 ||
rib_left_4 ||
rib_left_5 ||
rib_left_6 ||
rib_left_7 ||
rib_left_8 ||
rib_left_9 ||
rib_left_10 ||
rib_left_11 ||
rib_left_12 ||
rib_right_1 ||
rib_right_2 ||
rib_right_3 ||
rib_right_4 ||
rib_right_5 ||
rib_right_6 ||
rib_right_7 ||
rib_right_8 ||
rib_right_9 ||
rib_right_10 ||
rib_right_11 ||
rib_right_12 ||
scapula_left ||
scapula_right ||
clavicula_left | clavicle |
clavicula_right | clavicle |
hip_left | hip bone |
hip_right | hip bone |
sacrum ||
face ||
gluteus_maximus_left | gluteus maximus muscle |
gluteus_maximus_right | gluteus maximus muscle |
gluteus_medius_left | gluteus medius muscle |
gluteus_medius_right | gluteus medius muscle |
gluteus_minimus_left | gluteus minimus muscle |
gluteus_minimus_right | gluteus minimus muscle |
autochthon_left ||
autochthon_right ||
iliopsoas_left | iliopsoas muscle |
iliopsoas_right | iliopsoas muscle |
urinary_bladder ||
femur ||
patella ||
tibia ||
fibula ||
tarsal ||
metatarsal ||
phalanges_feet ||
humerus ||
ulna ||
radius ||
carpal ||
metacarpal ||
phalanges_hand ||
sternum ||
skull ||
subcutaneous_fat ||
skeletal_muscle ||
torso_fat ||
spinal_cord ||
