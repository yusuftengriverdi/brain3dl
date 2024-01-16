import numpy as np
from medpy.metric.binary import dc, hd, ravd


def compute_dice(prediction, reference) :
  for c in np.unique(reference) :
    dsc_val = dc(prediction == c, reference==c)
    # print(f'Dice coefficient class {c} equal to {dsc_val : .2f}')
  
  return dsc_val

def compute_hd(prediction, reference, voxel_spacing) :
  for c in np.unique(prediction) :
    hd_val = hd(prediction == c, reference==c, voxelspacing=voxel_spacing, connectivity=1)
    # print(f'Hausdorff distance class {c} equal to {hd_val : .2f}')
  return hd_val

def compute_ravd(prediction, reference) :
  for c in np.unique(prediction) :
    ravd_val = ravd(prediction == c, reference==c)
    # print(f'RAVD coefficient class {c} equal to {ravd_val : .2f}')

  return ravd_val
