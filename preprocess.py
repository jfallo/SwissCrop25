import argparse
import os
import json
import tarfile

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc


CL_DATA = ['cgdd', 'cgdd_seasonal']
P_DATA = ['parcel_id', 'n_px', 'mean_purity', 'overlay_flag', 'cell_idx', 'label']


def read_parquet_from_tar(tar, root, parquet_path, columns= None):
    """Read a Parquet file from a TAR archive."""
    member = tar.extractfile(f'{root}/{parquet_path}')
    if member is None:
        raise FileNotFoundError(f'Could not find {parquet_path} in TAR archive.')
    with member:
        return pd.read_parquet(member, columns= columns, engine= 'pyarrow')

def batch_parquet(tar, root, parquet_path, columns= None, batch_size= 2_000_000):
    member = tar.extractfile(f'{root}/{parquet_path}')
    if member is None:
        raise FileNotFoundError(f'Could not find {parquet_path} in TAR archive.')
    pf = pq.ParquetFile(member)
    yield from pf.iter_batches(batch_size= batch_size, columns= columns)

def arrow_to_numpy(arr, dtype= None):
    """Convert Arrow array to NumPy."""
    res = arr.to_numpy(zero_copy_only= False)
    return res if dtype is None else res.astype(dtype, copy= False)


def map_lnf_codes_to_labels(label_sheet_file, label_col= 'Crop_Label'):
    """Create mappings from label sheet."""

    # load label sheet file and remove excluded rows
    label_sheet = pd.read_excel(label_sheet_file)
    if 'Exclude' in label_sheet.columns:
        label_sheet = label_sheet[label_sheet['Exclude'] != True]

    # map labels to integer ids
    label_map = {label: i+1 for i, label in enumerate(label_sheet[label_col].unique())}
    num_classes = len(label_map) + 1

    # assign integers to LNF codes
    target_map = {0: 0, -1: 0}
    for _, row in label_sheet.iterrows():
        target_map[int(row['LNF_code'])] = label_map[row[label_col]]

    return label_map, target_map, num_classes


def generate_cgdd(year, window, cl, t_var= 'TabsD', t_base= 0.0, 
                  keys= ('climate_cell_e_epsg2056', 'climate_cell_n_epsg2056')):
    """Generate cumulative Growing Degree Day (CGDD) from daily climate data, aligned to MeteoSwiss 1 km grid for spatial climate analyses."""
    keys = list(keys)
    cl = cl.sort_values([*keys, 'date']).copy()
    cell = [cl[k] for k in keys]

    # calculate GDD
    cl['gdd'] = (cl[t_var] - t_base).clip(lower= 0.0)
    # accumulate GDD from window start
    cl['cgdd'] = cl['gdd'].groupby(cell).cumsum()
    # accumulate GDD by season, reseting from season starts
    starts = pd.to_datetime([window[0], f'{year - 1}-12-21', f'{year}-03-21', f'{year}-06-21', f'{year}-09-21']).to_numpy()
    season = pd.Series(np.searchsorted(starts, cl['date'].to_numpy(dtype= 'datetime64[ns]'), side= 'right'), index= cl.index)
    cl['cgdd_seasonal'] = cl['gdd'].groupby([*cell, season]).cumsum()
    
    return cl[[*keys, 'date', 'cgdd', 'cgdd_seasonal']]

def build_climate_lookup_table(cl, window, num_days, keys= ('climate_cell_e_epsg2056', 'climate_cell_n_epsg2056')):
    """Build climate data lookup table by date and MeteoSwiss 1 km grid cell."""
    keys = list(keys)
    cl = cl.sort_values([*keys, 'date']).copy()

    # index cells
    cells = cl[keys].drop_duplicates().reset_index(drop= True)
    cells['cell_idx'] = np.arange(len(cells))
    # map daily climate data to respective cells
    cell_idx = cl.groupby(keys, sort= False).ngroup().to_numpy()
    # flag border cells (cells on the Swiss national border may have no climate data)
    cells['has_cl_data'] = np.bincount(cell_idx, weights= cl[CL_DATA].isna().any(axis= 1).to_numpy(),
                                       minlength= len(cells)) == 0

    # index days from window start
    d_idx = (cl['date'] - pd.Timestamp(window[0])).dt.days.to_numpy()

    # build lookup table
    tab = np.full((len(cells), num_days, len(CL_DATA)), np.nan, dtype= np.float32)
    for i, cl_var in enumerate(CL_DATA):
        tab[cell_idx, d_idx, i] = cl[cl_var].to_numpy(dtype= np.float32)

    return tab, cells


def process_obs_batch(b, kept_ids, window, num_days):
    # map parcel ids to index in kept ids, -1 if parcel was dropped
    p_idx = arrow_to_numpy(
        pc.fill_null(pc.index_in(b.column('parcel_id'), value_set= kept_ids), -1),
        dtype= np.int32
    )

    # map dates to day index starting from window start
    dates = arrow_to_numpy(pc.cast(b.column('date'), pa.date32()))
    d_idx = (dates - np.datetime64(window[0])).astype(np.int16)


    # kept parcel identifier
    keep = (p_idx >= 0) & (d_idx >= 0) & (d_idx < num_days)

    return p_idx, keep, d_idx


def load_and_cache_year(tar, year, args, target_map, acq_types, s2_bands):
    out_path = os.path.join(args.out_dir, str(year))
    os.makedirs(out_path, exist_ok= True)
    os.makedirs(os.path.join(out_path, 'observations'), exist_ok= True)

    window = (f'{year - 1}-09-21', f'{year}-12-20')
    num_days = (pd.Timestamp(window[1]) - pd.Timestamp(window[0])).days + 1

    
    # ------ Daily climate data ------------------------------------ #

    # load climate_daily data in window
    cl_keys = ('climate_cell_e_epsg2056', 'climate_cell_n_epsg2056')
    cl = pd.concat((
        read_parquet_from_tar(tar, args.tar_root, f'climate_daily/{y}.parquet', columns= [*cl_keys, 'date', args.t_var]) 
        for y in (year - 1, year)), ignore_index= True
    )
    cl['date'] = pd.to_datetime(cl['date'])
    cl = cl[(cl['date'] >= window[0]) & (cl['date'] <= window[1])]

    # generate CGDD
    cgdd = generate_cgdd(year, window, cl[[*cl_keys, 'date', args.t_var]], args.t_var, args.t_base)
    cl = cl.merge(cgdd, on= [*cl_keys, 'date'])

    # build and save climate lookup table
    cl_tab, cells = build_climate_lookup_table(cl, window, num_days)
    np.save(os.path.join(out_path, 'cl_table.npy'), cl_tab)
    cells.to_parquet(os.path.join(out_path, 'cells.parquet'), index= False)


    # ------ Parcels ------------------------------------ #

    p = read_parquet_from_tar(tar, args.tar_root, f'parcels/{year}.parquet')
    # merge cells for climate lookup
    p = p.merge(cells, on= [*cl_keys], how= 'left', validate= 'm:1')
    assert p['cell_idx'].notna().all()
    # get labels
    p['label'] = p['lnf_code'].map(target_map).fillna(0).astype(np.int32)

    # drop parcels and reindex
    p = p[(p['drop_reason'] == 'none') & (p['n_px'] > 0)]
    if not args.keep_missing_climate_data:
        p = p[p['has_cl_data']]
    if not args.keep_missing_label:
        p = p[p['label'] > 0]
    p = p.reset_index(drop= True)
    kept_ids = pa.array(p['parcel_id'].to_numpy(), type= pa.string())  # already unique since parcel_id is unique
    num_parcels = len(p)

    p[P_DATA].to_parquet(os.path.join(out_path, 'parcels.parquet'), index= False)


    # ------ Observations ------------------------------------ #

    # get number of observations per parcel
    p_num_obs = np.zeros(num_parcels, dtype= np.int64)
    for b in batch_parquet(tar, args.tar_root, f'observations/{year}.parquet', ['parcel_id', 'date']):
        p_idx, keep, _ = process_obs_batch(b, kept_ids, window, num_days)
        p_num_obs += np.bincount(p_idx[keep], minlength= num_parcels)
    
    p_obs_start = np.concatenate(([0], np.cumsum(p_num_obs)), dtype= np.int64)
    num_obs = int(p_num_obs.sum())

    # initialize memory maps for storing observation data
    mm_d_idx = np.lib.format.open_memmap(os.path.join(out_path, 'observations/d_idx.npy'), mode= 'w+', 
                                         dtype= np.int16, shape= (num_obs,))
    mm_doy   = np.lib.format.open_memmap(os.path.join(out_path, 'observations/doy.npy'), mode= 'w+', 
                                         dtype= np.int16, shape= (num_obs,))
    mm_n_clr = np.lib.format.open_memmap(os.path.join(out_path, 'observations/n_clear_px.npy'), mode= 'w+', 
                                         dtype= np.int32, shape= (num_obs,))
    mm_bands = {acq_type: np.lib.format.open_memmap(os.path.join(out_path, f'observations/s2_bands_{acq_type}.npy'), mode= 'w+', 
                                                    dtype= np.float32, shape= (num_obs, len(args.bands))) 
                for acq_type in acq_types}
    mms = (mm_d_idx, mm_doy, mm_n_clr, *mm_bands.values())

    # track number of observations written so far per parcel
    p_obs_written = np.zeros(num_parcels, dtype= np.int64)
    # fill memory maps
    for b in batch_parquet(tar, args.tar_root, f'observations/{year}.parquet', ['parcel_id', 'date', 'doy', 'n_clear_px'] + s2_bands):
        p_idx, keep, d_idx = process_obs_batch(b, kept_ids, window, num_days)
        
        if not keep.any():
            continue
        p_idx = p_idx[keep]

        # get parcel index locations after stable sorting parcel index: [0, 2, 1, 0, 0, 2] -> [0, 3, 4, 2, 1, 5]
        b_sort_locs = np.argsort(p_idx, kind= 'stable')
        # stable sort parcel index
        b_p = p_idx[b_sort_locs]

        # location of first instance for each parcel index
        starts = np.concatenate(([0], np.flatnonzero(np.diff(b_p)) + 1))
        # number of observations for each parcel index
        lens = np.diff(np.concatenate((starts, [len(b_p)])))
        # position of observations within respective parcel index: [0, 0, 0, 1, 2, 2] -> [0, 1, 2, 0, 0, 1]
        local_pos = np.arange(len(b_p)) - np.repeat(starts, lens)

        # get memory-map write locations
        write_locs = p_obs_start[b_p] + p_obs_written[b_p] + local_pos
        # write to memory-maps
        mm_d_idx[write_locs] = d_idx[keep][b_sort_locs]
        mm_doy[write_locs] = arrow_to_numpy(b.column('doy'), dtype= np.int32)[keep][b_sort_locs]
        mm_n_clr[write_locs] = arrow_to_numpy(b.column('n_clear_px'), dtype= np.int32)[keep][b_sort_locs]
        for acq_type in acq_types:
            mm_bands[acq_type][write_locs] = np.stack([
                arrow_to_numpy(b.column(f's2_{band}_{acq_type}'), dtype= np.float32)
                for band in args.bands
            ], axis= 1)[keep][b_sort_locs]
        
        # update observations written per parcel
        p_obs_written[b_p[starts]] += lens
    
    assert (p_obs_written == p_num_obs).all(), "Missing observations in write."
    for mm in mms:
        mm.flush()
    np.save(os.path.join(out_path, 'observations/parcel_starts.npy'), p_obs_start)

    meta = {
        'year': year,
        'window': window,
        'num_days': num_days,
        'num_parcels': num_parcels,
        'num_obs': num_obs,
        'num_cells': len(cells)
    }
    with open(os.path.join(out_path, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent= 2)


def main(args):
    label_map, target_map, num_classes = map_lnf_codes_to_labels(args.label_sheet, args.label_col)
    with open(os.path.join(args.out_dir, 'label_map.json'), 'w') as f:
        json.dump(label_map, f)

    acq_types = ['clear'] if args.only_clear_px else ['clear', 'all']
    s2_bands = [f's2_{band}_{acq_type}' for band in args.bands for acq_type in acq_types]
    with tarfile.open(args.tar, 'r') as tar:
        for year in args.years:
            load_and_cache_year(tar, year, args, target_map, acq_types, s2_bands)

    meta = {
        'years': args.years,
        'labels': {
            'source_col': args.label_col,
            'background_label': 0,
            'num_classes': num_classes
        },
        'filters': {
            'keep_missing_climate_data': bool(args.keep_missing_climate_data),
            'keep_missing_label': bool(args.keep_missing_label)
        },
        'acq_types': acq_types,
        'bands': args.bands,
        't_var': args.t_var,
        't_base': args.t_base,
        'cl_data': CL_DATA,
        'p_data': P_DATA
    }
    with open(os.path.join(args.out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent= 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tar', type= str, default= f'data/SwissCrop25-ParcelTS.tar', 
                        help= "Path to SwissCrop25 TAR archive.")
    parser.add_argument('--tar_root', type= str, default= 'SwissCrop25-ParcelTS')
    parser.add_argument('--label_sheet', type= str, default= f'data/SwissCrop25_label_sheet.xlsx',
                        help= "Path to SwissCrop25 label sheet file.")
    parser.add_argument('--label_col', type= str, default= 'Crop_Label',
                        help= "Column in label sheet file for describing crop labels.")
    parser.add_argument('--years', type= int, nargs= '+', default= list(range(2019, 2026)))
    parser.add_argument('--t_var', type= str, default= 'TabsD',
                        help= "Temperature reference variable.")
    parser.add_argument('--t_base', type= float, default= 0.0,
                        help= "Base temperature for computing GDD.")
    parser.add_argument('--keep_missing_climate_data', action= 'store_true',
                        help= "Keep parcels in grid cells with missing climate data.")
    parser.add_argument('--keep_missing_label', action= 'store_true',
                        help= "Keep parcels with missing crop label.")
    parser.add_argument('--bands', type= str, nargs= '+', 
                        default= ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12'])
    parser.add_argument('--only_clear_px', action= 'store_true',
                        help= "Only include cloud-free observations.")
    parser.add_argument('--out_dir', type= str, default= 'data/processed/',
                        help= "Output directory for caches.")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok= True)
    main(args)
