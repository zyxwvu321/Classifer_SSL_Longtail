# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 18:40:40 2020
dicom info output, save csv for ISIC20
@author: cmj
"""

import pydicom

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
fd_dcm = '../data/ISCI20/train_dcm'
#fd_dcm = 'D:/tmp'

out_csv = './dat/isic20_dcm_meta.csv'
fns_dcm = sorted(list(Path(fd_dcm).glob('*.dcm')))

info_dcms = []

for fn in tqdm(fns_dcm):
    

    
    with pydicom.dcmread(fn) as dc:
        
        hh,ww = dc.Rows,dc.Columns
        ImageType = str(dc.ImageType)
        StudyDate = dc.StudyDate
        ContentDate  = dc.ContentDate
        StudyTime = dc.StudyTime 
        ContentTime  = dc.ContentTime  
        AccessionNumber = dc.AccessionNumber
        Modality = dc.Modality
        Manufacturer = dc.Manufacturer
        InstitutionName  = dc.InstitutionName 
        CodeValue  = dc.AnatomicRegionSequence[0].CodeValue 
        CodingSchemeDesignator  = dc.AnatomicRegionSequence[0].CodingSchemeDesignator
        CodeMeaning = dc.AnatomicRegionSequence[0].CodeMeaning
        PatientSex = dc.PatientSex
        PatientAge = dc.PatientAge
        BodyPartExamined  = dc.BodyPartExamined
        StudyInstanceUID  = dc.StudyInstanceUID
        SeriesInstanceUID = dc.SeriesInstanceUID
        StudyID = dc.StudyID
        SeriesNumber  = dc.SeriesNumber
        InstanceNumber= dc.InstanceNumber 
        PatientOrientation = dc.PatientOrientation
        SamplesPerPixel = dc.SamplesPerPixel
        PhotometricInterpretation = dc.PhotometricInterpretation
        PlanarConfiguration = dc.PlanarConfiguration
        BitsStored  = dc.BitsStored
        PixelRepresentation = dc.PixelRepresentation
        BurnedInAnnotation = dc.BurnedInAnnotation
        LossyImageCompression = dc.LossyImageCompression
        
        meta = [hh,ww,ImageType,StudyDate,ContentDate,StudyTime,ContentTime,AccessionNumber,Modality, \
                Manufacturer,InstitutionName,CodeValue,CodingSchemeDesignator,CodeMeaning,PatientSex,\
                PatientAge,BodyPartExamined,StudyInstanceUID,SeriesInstanceUID,StudyID,SeriesNumber,\
                InstanceNumber,PatientOrientation,SamplesPerPixel,PhotometricInterpretation,PlanarConfiguration,\
                BitsStored,PixelRepresentation,BurnedInAnnotation,LossyImageCompression]
        
        info_dcms.append(np.array(meta))

info_dcms = np.array(info_dcms)


fn_stem = [fn.stem for fn in fns_dcm]

columns = ['hh','ww','ImageType','StudyDate','ContentDate','StudyTime','ContentTime','AccessionNumber','Modality',' \
                Manufacturer','InstitutionName','CodeValue','CodingSchemeDesignator','CodeMeaning','PatientSex','\
                PatientAge','BodyPartExamined','StudyInstanceUID','SeriesInstanceUID','StudyID','SeriesNumber','\
                InstanceNumber','PatientOrientation','SamplesPerPixel','PhotometricInterpretation','PlanarConfiguration','\
                BitsStored','PixelRepresentation','BurnedInAnnotation','LossyImageCompression']
        

df = pd.DataFrame(data = info_dcms, index = fn_stem,columns = columns)
df.to_csv(out_csv, index=fn_stem)
