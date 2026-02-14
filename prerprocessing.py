import SimpleITK as stk
import numpy as np
import os
import json

def load_dicom_series(file):
    reader = stk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(file)

    if not series_ids:
        raise RuntimeError(f"No DICOM series found in directory: {file}")
    
    series_files = reader.GetGDCMSeriesFileNames(file, series_ids[0])
    reader.SetFileNames(series_files)
    image = reader.Execute()
    return image

def standardize_orientation(image):
    # RAI = Right-Anterior-Inferior (common standard)
    orient_filter = stk.OrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation("RAI")
    return orient_filter.Execute(image)

def resample_image(image, out_spacing=(1.0, 1.0, 1.0)):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    #Add gaussian low pass filter to reduce noise before resampling

    out_size = [
        int(round(original_size[i] * original_spacing[i] / out_spacing[i]))
        for i in range(3)
    ]

    resampler = stk.ResampleImageFilter()
    resampler.SetOutputSpacing(out_spacing)
    resampler.SetSize(out_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetInterpolator(stk.sitkBSpline)

    return resampler.Execute(image)

def bias_field_correction(image):
    # Convert to float for N4 bias field correction
    image = stk.Cast(image, stk.sitkFloat32)
    mask = stk.OtsuThreshold(image, 0, 1, 255)
    corrector = stk.N4BiasFieldCorrectionImageFilter()
    corrected = corrector.Execute(image, mask)
    return corrected

def Normalization(image):
    return stk.Normalize(image)


def denoise_image(image):
    return stk.CurvatureFlow(
        image,
        timeStep=0.125,
        numberOfIterations=5
    )

def center_crop(image, crop_size=(128, 256, 256)):
    size = image.GetSize()
    
    # Ensure crop_size doesn't exceed image size
    actual_crop_size = [
        min(crop_size[i], size[i])
        for i in range(3)
    ]
    
    start = [
        max(0, (size[i] - actual_crop_size[i]) // 2)
        for i in range(3)
    ]

    crop_filter = stk.RegionOfInterestImageFilter()
    crop_filter.SetSize(actual_crop_size)
    crop_filter.SetIndex(start)

    return crop_filter.Execute(image)