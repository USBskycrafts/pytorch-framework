import numpy as np
from skimage.measure import label, regionprops, find_contours
from typing import List, Tuple, Union


def mask_to_border(mask):
    # Convert a mask to border image
    mask = np.squeeze(mask, axis=0)
    h, w = mask.shape
    border = np.zeros((h, w))

    contours = find_contours(mask, 0.5)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255

    return border


""" Mask to bounding boxes """


def mask_to_bbox(mask) -> Union[Tuple, None]:
    bboxes = None
    mask = mask_to_border(mask)
    l = label(mask)
    for region in regionprops(l):
        if region.area >= 40:
            minr, minc, maxr, maxc = region.bbox
            if bboxes is None:
                bboxes = (np.array(region.centroid), maxc - minc, maxr - minr)
            elif bboxes[1] * bboxes[2] < region.area:
                bboxes = (np.array(region.centroid), maxc - minc, maxr - minr)
    return bboxes
