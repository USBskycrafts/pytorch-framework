import dataset.BraTS2020.NIFTI1Loader as BraTS2020
import dataset.BraTS2021.NIFTI1Loader as BraTS2021

dataset_list = {
    "BraTS2020": BraTS2020.NIFTI1Loader,
    "BraTS2021": BraTS2021.NIFTI1Loader
}
