import json
import argparse
import warnings
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')


# taxonomy group → display order and colour
GROUP_ORDER  = ['Arable Land', 'Grassland', 'Permanent', 'Forest',
                'Water', 'Unproductive Area', 'Built-up', 'Wetland']
GROUP_COLORS = {
    'Arable Land':       '#4e9a50',
    'Grassland':         '#a8d08d',
    'Permanent':         '#f4a460',
    'Forest':            '#8b6914',
    'Water':             '#5b9bd5',
    'Unproductive Area': '#bfbfbf',
    'Built-up':          '#e06c75',
    'Wetland':           '#85c1e9',
}


def load_class_info(cache_dir, label_sheet, label_col):
    df = pd.read_excel(label_sheet)
    if 'Exclude' in df.columns:
        df = df[df['Exclude'] != True]
    df = df.drop_duplicates(subset= label_col, keep= 'first').reset_index(drop= True)

    # get label index from label map
    with open(os.path.join(cache_dir, 'label_map.json')) as f:
        label_map = json.load(f)
    df['class_idx'] = df[label_col].astype(str).map(label_map).astype(int)

    df['Crop_Label_lv3'] = df['Crop_Label_lv3'].fillna(df['Crop_Label_lv2'])

    return df


def sort_order(df):
    """Return list of class indices sorted by lv3 (GROUP_ORDER) → lv2 → lv1 → LNF_code."""
    rows = []
    for grp in GROUP_ORDER:
        sub = df[df['Crop_Label_lv3'] == grp].sort_values(['Crop_Label_lv2', 'Crop_Label_lv1', 'LNF_code'])
        rows.append(sub)
    ordered = pd.concat(rows, ignore_index= True)
    
    return ordered['class_idx'].tolist(), ordered


def group_boundaries(df):
    """Return list of (start, end, group_name) for divider lines."""
    bounds, prev_grp, start = [], None, 0
    for i, grp in enumerate(df['Crop_Label_lv3']):
        if grp != prev_grp:
            if prev_grp is not None:
                bounds.append((start, i, prev_grp))
            start, prev_grp = i, grp
    bounds.append((start, len(df), prev_grp))
    
    return bounds


def row_normalise(cm, idx):
    """Recall (%) over classes in idx; rows are divided by the full row sum. Unsupported rows -> NaN."""
    sub = cm[np.ix_(idx, idx)]
    row_sums = cm[idx].sum(axis= 1, keepdims= True)
    out = np.divide(sub, row_sums, out= np.full_like(sub, np.nan), where= row_sums > 0) * 100
    return out




TICK_FS  = 12.0  # class label fontsize
TITLE_FS = 16
AXIS_FS  = 13


def _draw_matrix(ax, cm_n, class_labels, bounds):
    """Draw heatmap with grid, group bands, and tick labels. Returns AxesImage."""
    n = len(class_labels)
    im = ax.imshow(cm_n, aspect= 'equal', interpolation= 'nearest',
                   cmap= 'Blues', vmin= 0, vmax= 100)

    # cell-level grid
    ax.set_xticks(np.arange(-0.5, n, 1), minor= True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor= True)
    ax.grid(which= 'minor', color= 'gray', linewidth= 0.2, alpha= 0.3)
    ax.tick_params(which= 'minor', length= 0)

    # thick group boundary lines
    for start, end, grp in bounds:
        for pos in [start - 0.5, end - 0.5]:
            ax.axhline(pos, color= 'white', linewidth= 1.2, alpha= 0.85)
            ax.axvline(pos, color= 'white', linewidth= 1.2, alpha= 0.85)

    # colour bands on left spine
    for start, end, grp in bounds:
        ax.add_patch(mpatches.Rectangle(
            (-3.0, start - 0.5), 1.5, end - start,
            transform= ax.transData, clip_on= False,
            color= GROUP_COLORS.get(grp, '#cccccc'), linewidth= 0))

    ax.set_title("Confusion Matrix", fontsize= TITLE_FS, fontweight= 'bold', pad= 7)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))

    tick_labels = [l[:16] for l in class_labels]
    ax.set_xticklabels(tick_labels, rotation= 90, fontsize= TICK_FS, ha= 'center', va= 'top')
    ax.set_yticklabels(tick_labels, fontsize= TICK_FS)

    ax.set_xlabel('Predicted', fontsize= AXIS_FS, labelpad= 4)
    ax.set_ylabel('Ground truth', fontsize= AXIS_FS, labelpad= 4)

    return im


def make_single_figure(cm_n, df, class_labels):
    """Individual full-width figure for one model."""
    bounds = group_boundaries(df)

    # figsize chosen so axes are square:
    # ax_width_in  = 0.80 × 18 = 14.40
    # ax_height_in = 0.740 × 19 = 14.06  → no right-side whitespace
    # left/bottom margins larger to fit 12pt tick labels
    fig = plt.figure(figsize= (18,19))
    ax = fig.add_axes([0.15, 0.23, 0.80, 0.740])

    im = _draw_matrix(ax, cm_n, class_labels, bounds)

    # narrow colorbar bottom-left; leave visible gap below the matrix
    cax = fig.add_axes([0.15, 0.11, 0.24, 0.018])
    cbar = fig.colorbar(im, cax= cax, orientation= 'horizontal')
    cbar.set_label('Recall (%)', fontsize= AXIS_FS)
    cbar.ax.tick_params(labelsize= 11)

    # group legend to the right of the colorbar
    legend_patches = [mpatches.Patch(color= GROUP_COLORS[g], label= g)
                      for g in GROUP_ORDER if g in GROUP_COLORS]
    fig.legend(handles= legend_patches, fontsize= 13,
               bbox_to_anchor= (0.96,0.135), loc= 'upper right',
               ncol= 2, framealpha= 0.85, handlelength= 1.4, columnspacing= 1.2)

    return fig


def main():
    parser = argparse.ArgumentParser(description= "Confusion matrix figures from train.py outputs.")
    parser.add_argument('--run', required= True,
                        help= "Visualizes the confusion matrix for given run.")
    parser.add_argument('--split', choices= ['val', 'test'], default= 'test')
    parser.add_argument('--px', action= 'store_true', 
                        help= "Use pixel-weighted confusion matrix.")
    parser.add_argument('--cache_dir', default= 'data/cache', 
                        help= "Location of label_map.json.")
    parser.add_argument('--label_sheet', default= 'data/SwissCrop25_label_sheet.xlsx')
    parser.add_argument('--label_col', default= 'Crop_Label')
    args = parser.parse_args()

    
    df = load_class_info(args.cache_dir, args.label_sheet, args.label_col)
    idx_order, ordered_df = sort_order(df)
    class_labels = ordered_df[args.label_col].tolist()

    fname = f"{args.split}_confusion_matrix{'_px' if args.px else ''}"
    
    cm = np.load(os.path.join(args.run, f'{fname}.npy')).astype(np.float64)
    cm = row_normalise(cm, idx_order)

    print(f"Average recall: {np.nanmean(np.diag(cm)):.1f}%")
    fig = make_single_figure(cm, ordered_df, class_labels)
    out_path = os.path.join(args.run, f'{fname}.pdf')
    fig.savefig(out_path, dpi= 200, bbox_inches= 'tight')
    plt.close(fig)




if __name__ == "__main__":
    main()
