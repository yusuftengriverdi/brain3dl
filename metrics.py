import numpy as np
from medpy.metric.binary import hd, ravd

def haussdorf(gt, pred, voxelspacing = (9.375e-01, 9.375e-01, 1.5)):
    """Compute relative absolute volume difference across classes. The corresponding labels should be
    previously matched.
    Args:
        gt (np.ndarray): Grounth truth
        pred (np.ndarray): Labels
        voxelspacing (tuple): voxel_spacing
    Returns:
        list: Dice scores per tissue [CSF, GM, WM]
    """
    gt, pred = gt.detach().numpy(), pred.detach().numpy()
    classes = np.unique(gt[gt != 0]).astype(int)
    glob_hd_values = []
    print(gt.shape, pred.shape)
    for gt_, pred_ in zip(gt, pred):
        hd_values = np.zeros((len(classes)))
        pred_ = np.argmax(pred_, axis=0)
        # print("EXCUSE MOI?? ------------------------", np.unique(pred_, return_counts=True), pred_.shape)
        gt_ = np.argmax(gt_, axis=0)
        # print("EXCUSE MOI?? ------------------------", np.unique(gt_, return_counts=True), gt_.shape)
        for i in classes:
            bin_pred = np.where(pred_ == i, 1, 0)
            bin_gt = np.where(gt_ == i, 1, 0)
            #try:
            hd_values[i-1] = hd(bin_pred, bin_gt, voxelspacing=voxelspacing)
            #except:
                #hd_values[i-1] = np.nan
            glob_hd_values.append(hd_values.mean())
    return glob_hd_values.mean()

def avd(gt: np.ndarray, pred: np.ndarray, voxelspacing: tuple):
    """Compute relative absolute volume difference across classes. The corresponding labels should be
    previously matched.
    Args:
        gt (np.ndarray): Grounth truth
        pred (np.ndarray): Labels
        voxelspacing (tuple): voxel_spacing
    Returns:
        list: Dice scores per tissue [CSF, GM, WM]
    """
    gt, pred = gt.detach().numpy(), pred.detach().numpy()
    classes = np.unique(gt[gt != 0]).astype(int)
    avd = np.zeros((len(classes)))
    for i in classes:
        bin_pred = np.where(pred == i, 1, 0)
        bin_gt = np.where(gt == i, 1, 0)
        vol_pred = np.count_nonzero(bin_pred)
        vol_gt = np.count_nonzero(bin_gt)
        unit_volume = voxelspacing[0] * voxelspacing[1] * voxelspacing[2]
        avd[i-1] = np.abs(vol_pred - vol_gt) * unit_volume
    return avd.mean()


def rel_abs_vol_dif(gt: np.ndarray, pred: np.ndarray):
    """Compute relative absolute volume difference across classes. The corresponding labels should be
    previously matched.
    Args:
        gt (np.ndarray): Grounth truth
        pred (np.ndarray): Labels
    Returns:
        list: Dice scores per tissue [CSF, GM, WM]
    """
    gt, pred = gt.detach().numpy(), pred.detach().numpy()    
    classes = np.unique(gt[gt != 0]).astype(int)
    ravd_values = np.zeros((len(classes)))
    for i in classes:
        bin_pred = np.where(pred == i, 1, 0)
        bin_gt = np.where(gt == i, 1, 0)
        try:
            ravd_values[i-1] = ravd(bin_gt, bin_pred)
        except:
            ravd_values[i-1] = np.nan
    return ravd_values.mean()