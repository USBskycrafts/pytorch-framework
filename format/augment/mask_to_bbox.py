import numpy as np
from skimage.measure import label, regionprops, find_contours
from typing import List, Tuple


def mask_to_border(mask):
    # Convert a mask to border image
    mask = np.squeeze(mask, axis=0)
    h, w = mask.shape
    border = np.zeros((h, w))

    contours = find_contours(mask, 1)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255

    return border


""" Mask to bounding boxes """


def mask_to_bbox(mask) -> List[Tuple[Tuple[int, ...], int, int]]:
    bboxes = []

    mask = mask_to_border(mask)
    l = label(mask)
    for region in regionprops(l):
        if region.area >= 40:
            minr, minc, maxr, maxc = region.bbox
            print(f"{minr}, {minc}, {maxr}, {maxc} as bbox")
            bboxes.append(((minc, minr), maxc - minc, maxr - minr))
    return bboxes
