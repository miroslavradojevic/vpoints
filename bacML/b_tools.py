import os
import random
import numpy as np


def load_data(data_dir, img_ext, annot_dirs):
    ftifs = [os.path.join(data_dir, f)
             for f in os.listdir(data_dir)
             if f.endswith("." + img_ext)]

    flist = []
    for ftif in ftifs:
        ftif_dir = os.path.abspath(os.path.join(ftif, os.pardir))
        ftif_name = os.path.basename(ftif)
        flist.append((ftif,))
        for annot_dir in annot_dirs:
            flist[-1] = flist[-1] + (os.path.join(ftif_dir, annot_dir, ftif_name),)
    return flist


def systematic_resampling(weights, mask, number_samples):
    cws = np.zeros(len(weights))
    for i in range(len(cws)):
        if mask is None:
            cws[i] = weights[i] + (0.0 if (i == 0) else cws[i - 1])
        else:
            cws[i] = (weights[i] if (mask[i] == 255) else 0.0) + (0.0 if (i == 0) else cws[i - 1])
    out = np.zeros(number_samples).astype(int)
    totalmass = cws[len(cws) - 1]
    # systematic re-sampling
    i = int(0)
    u1 = (totalmass / float(number_samples)) * random.uniform(0, 1)
    for j in range(number_samples):
        uj = u1 + j * (totalmass / float(number_samples))
        while uj > cws[i]:
            i += 1
        out[j] = i
    return out


def grid_sampling(mask, step):
    rows = []
    cols = []
    for r in range(0, mask.shape[0], step):
        for c in range(0, mask.shape[1], step):
            if mask[r, c] > 0:
                rows.append(r)
                cols.append(c)

    return rows, cols


def updatecat(category_name, category_index, map):
    if category_name not in map:
        category_index += 1
        map[category_name] = category_index
    else:
        category_index = map[category_name]
    print(category_index, " -> ", map)
    return category_index, map


def get_locs(n_rows, n_cols, step, D_rows, D_cols, circ_radius_ratio):
    rows = []
    cols = []
    for row in range(0, n_rows, step):
        for col in range(0, n_cols, step):
            row0 = int(row - D_rows / 2)
            row1 = int(row + D_rows / 2)
            col0 = int(col - D_cols / 2)
            col1 = int(col + D_cols / 2)
            if row0 >= 0 and row1 < n_rows and col0 >= 0 and col1 < n_cols:
                if circ_radius_ratio is not None:
                    if pow(row - n_rows / 2, 2) + pow(col - n_cols / 2, 2) <= pow(circ_radius_ratio * min(n_rows / 2, n_cols / 2), 2):
                        rows.append(row)
                        cols.append(col)
                else:
                    rows.append(row)
                    cols.append(col)

    return rows, cols


def testfun(a=(1, 2, 3), b=2):
    c = 0
    for i in range(0, len(a)):
        c += a[i] * b
    return c
