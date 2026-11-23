import os
import json

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class ParcelDataset(Dataset):
    def __init__(
        self,
        cache_dir, years, 
        acq_type= 'clear', bands= ('B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12'), 
        temp_var= 'cgdd',
        band_stats= None, time_stats= None,
        ignore_index= None,
        min_px= 1, min_purity= 0.0, drop_overlay= False, drop_invalid= True,
        min_clear_frac= 0.0,
        max_seq_len= 24, use_temp_calendar= False,
        use_temp_subsampling= False,    # sliding window when false
        use_random_subsampling= False,
        normalize_timestamps= True,
        seed= 42, training= False, verbose= False
    ):
        self.cache_dir            = cache_dir
        self.years                = self._ensure_list(years)
        self.acq_type             = acq_type
        self.bands                = bands
        self.temp_var             = temp_var
        self.band_stats           = band_stats
        self.time_stats           = time_stats
        self.normalize_timestamps = normalize_timestamps
        self.ignore_index         = ignore_index
        # parcel filters
        self.min_px               = min_px
        self.min_purity           = min_purity
        self.drop_overlay         = drop_overlay
        self.drop_invalid         = drop_invalid
        # observation filters
        self.min_clear_frac       = min_clear_frac
        # time dimension setup
        self.max_seq_len          = max_seq_len
        self.use_temp_calendar    = use_temp_calendar
        # sampling methods
        self.use_temp_subsampling   = use_temp_subsampling
        self.use_random_subsampling = use_random_subsampling
        # misc
        self.seed                 = seed
        self.training             = training
        self.verbose              = verbose
        self._handles             = {}  # references to open memory-mapped arrays


        # load metadata from cache dir
        with open(os.path.join(cache_dir, 'meta.json')) as f:
            meta = json.load(f)

        # availability check
        for y in self.years:
            assert y in meta['years'], f"Year {y} not in cached data."
        assert acq_type in meta['acq_types'], f"Acquisition type {acq_type} not in cached data."
        for band in bands:
            assert band in meta['bands'], f"Band {band} not in cached data."
        assert self.temp_var in meta['cl_data'], f"Temperature variable {self.temp_var} not in cached data."
        if normalize_timestamps and time_stats is None and not training:
            raise ValueError("normalize_timestamps=True requires time_stats (compute them on the training dataset).")

        self.num_classes = meta['labels']['num_classes']
        # band positions in band value arrays
        self.band_locs = np.array([meta['bands'].index(band) for band in bands])
        # temperature variable position in cell climate data array
        self.temp_var_loc = meta['cl_data'].index(self.temp_var)
        
        # global indexing by (year, parcel)
        g_y = []
        g_p = []
        # load yearly data from cache dir
        self.y_data = []
        for i, y in enumerate(self.years):
            y_dir = os.path.join(self.cache_dir, str(y))
            with open(os.path.join(y_dir, 'meta.json')) as f:
                y_meta = json.load(f)
            assert y_meta['year'] == y, "Year mismatch."

            # load parcels
            p = pd.read_parquet(os.path.join(y_dir, 'parcels.parquet'), columns= meta['p_data'])
            p_obs_starts = np.load(os.path.join(y_dir, 'observations/parcel_starts.npy'))

            # filter parcels
            keep = (p['n_px'] >= min_px) & (p['mean_purity'].fillna(0.0) >= min_purity)
            if drop_overlay:
                keep &= ~p['overlay_flag'].astype(bool)
            rows = np.flatnonzero(keep.to_numpy())
            if drop_invalid:
                rows = rows[self._count_valid_obs(y_dir, rows, p['n_px'].to_numpy(dtype= np.int64), p_obs_starts) > 0]

            p = p.iloc[rows]
            p_ids = p['parcel_id'].to_numpy()
            p_n_pxs = p['n_px'].to_numpy(dtype= np.int64)
            p_cell_idxs = p['cell_idx'].to_numpy(dtype= np.int64)
            p_labels = p['label'].to_numpy(dtype= np.int64)
            

            # store year
            self.y_data.append({
                'dir': y_dir,
                'meta': y_meta,
                'p': {
                    'rows': rows,
                    'ids': p_ids,
                    'n_pxs': p_n_pxs,
                    'cell_idxs': p_cell_idxs,
                    'labels': p_labels,
                    'obs_starts': p_obs_starts
                }
            })

            g_y.append(np.full(len(rows), i, dtype= np.int32))
            g_p.append(np.arange(len(rows), dtype= np.int64))
        
        self._g_y = np.concatenate(g_y)
        self._g_p = np.concatenate(g_p)

        
        if training:
            # compute normalization statistics
            if band_stats is None:
                self.band_stats = self.compute_band_stats()
            if normalize_timestamps and time_stats is None:
                self.time_stats = self.compute_time_stats()
        

    @staticmethod
    def _ensure_list(x):
        return x if isinstance(x, (list, tuple)) else [x]


    def _check_obs_valid(self, clear_frac, n_clear_px):
        """Check observations meet the minimum required number and fraction of clear pixels."""
        valid = clear_frac >= self.min_clear_frac
        if self.acq_type == 'clear':
            valid &= n_clear_px > 0
        return valid

    def _count_valid_obs(self, y_dir, rows, n_px, p_obs_starts, chunk_size= 50_000):
        """Number of usable observations per kept parcel (clear pixels, minimum clear fraction)."""
        y_obs_dir = os.path.join(y_dir, 'observations')
        n_clear_px = np.load(os.path.join(y_obs_dir, 'n_clear_px.npy'), mmap_mode= 'r')
 
        counts = np.zeros(len(rows), dtype= np.int64)
        for a in range(0, len(rows), chunk_size):
            r = rows[a:a + chunk_size]
            r_lo, r_hi = int(r[0]), int(r[-1])
            lo, hi = int(p_obs_starts[r_lo]), int(p_obs_starts[r_hi + 1])
 
            chunk_n_px = np.repeat(n_px[r_lo:r_hi + 1], np.diff(p_obs_starts[r_lo:r_hi + 2]))
            chunk_n_clear_px = np.asarray(n_clear_px[lo:hi])
            chunk_clear_frac = chunk_n_clear_px / np.maximum(chunk_n_px, 1)
            
            valid = self._check_obs_valid(chunk_clear_frac, chunk_n_clear_px)
            num_valid = np.concatenate(([0], np.cumsum(valid)))
            
            counts[a:a + chunk_size] = num_valid[p_obs_starts[r+1] - lo] - num_valid[p_obs_starts[r] - lo]
 
        return counts


    def __len__(self):
        return len(self._g_p)

    def __getstate__(self):
        state = self.__dict__.copy()
        # workers load only memory-mapped arrays relevant to themselves
        state['_handles'] = {}
        return state

    def _open(self, i):
        h = self._handles.get(i)
        if h is None:
            y_dir = self.y_data[i]['dir']
            y_obs_dir = os.path.join(y_dir, 'observations')
            
            h = {
                'd_idx': np.load(os.path.join(y_obs_dir, 'd_idx.npy'), mmap_mode= 'r'),
                'doy': np.load(os.path.join(y_obs_dir, 'doy.npy'), mmap_mode= 'r'),
                'n_clear_px': np.load(os.path.join(y_obs_dir, 'n_clear_px.npy'), mmap_mode= 'r'),
                'band_vals': np.load(os.path.join(y_obs_dir, f's2_bands_{self.acq_type}.npy'), mmap_mode= 'r'),
                'cl_table': np.load(os.path.join(y_dir, 'cl_table.npy'), mmap_mode= 'r')
            }
            
            self._handles[i] = h

        return h


    def _sliding_window(self, idx, clear_frac):
        """Clearest valid observations per K time segments of equal number of observations."""
        K = self.max_seq_len
        segments = np.array_split(idx, min(K, len(idx)))
        obs = np.array([segment[np.argmax(clear_frac[segment])] for segment in segments], dtype= np.int64)
        return obs, np.arange(len(obs), dtype= np.int64)

    def _T3S(self, idx, cgdd, clear_frac):
        """K equal-width bins over the parcel's own CGDD range, choosing the clearest observation per bin."""
        K = self.max_seq_len 
        lo, hi = cgdd[idx].min(), cgdd[idx].max()
        if hi > lo:
            bins = np.minimum(((cgdd[idx] - lo) / (hi - lo) * K).astype(np.int64), K-1)
        else:
            bins = np.zeros(len(idx), dtype= np.int64)
        by_clear_frac = np.lexsort((-clear_frac[idx], bins))
        bins = bins[by_clear_frac]
        _, first = np.unique(bins, return_index=True)
        return idx[by_clear_frac[first]], bins[first]

    def _get_obs(self, valid, cgdd, clear_frac):
        valid = np.flatnonzero(valid)
        if len(valid) == 0:
            return valid, valid
        elif len(valid) <= self.max_seq_len:
            return valid, np.arange(len(valid), dtype= np.int64)

        if self.use_random_subsampling and self.training:
            obs = np.sort(np.random.choice(valid, self.max_seq_len, replace= False))
            loc = np.arange(len(obs), dtype= np.int64)
        else:
            if self.use_temp_subsampling:
                obs, loc = self._T3S(valid, cgdd, clear_frac)
            else:
                obs, loc = self._sliding_window(valid, clear_frac)

        return obs, loc

    def __getitem__(self, j):
        y_idx = int(self._g_y[j])
        p_idx = int(self._g_p[j])

        y_parcels = self.y_data[y_idx]['p']
        data = self._open(y_idx)

        row = y_parcels['rows'][p_idx]
        start, end = y_parcels['obs_starts'][row], y_parcels['obs_starts'][row + 1]

        d_idx = data['d_idx'][start:end]
        band_vals = data['band_vals'][start:end][:, self.band_locs]
        n_px = y_parcels['n_pxs'][p_idx]
        doy = data['doy'][start:end]
        n_clear_px = data['n_clear_px'][start:end]
        clear_frac = n_clear_px / np.maximum(n_px, 1)
        cgdd = None
        if self.use_temp_calendar or self.use_temp_subsampling:
            cell_idx = y_parcels['cell_idxs'][p_idx]
            cgdd = data['cl_table'][cell_idx, d_idx, self.temp_var_loc]

        # filter observations
        valid = self._check_obs_valid(clear_frac, n_clear_px)
        valid &= ~np.isnan(band_vals).any(axis= 1)
        if cgdd is not None:
            valid &= ~np.isnan(cgdd)

        # get observations
        obs, loc = self._get_obs(valid, cgdd, clear_frac)

        if self.use_temp_calendar:
            obs_timestamps = cgdd[obs]
        else:
            obs_timestamps = doy[obs]
        obs_timestamps = obs_timestamps.astype(np.float32)
        if self.normalize_timestamps and self.time_stats is not None:
            obs_timestamps = (obs_timestamps - self.time_stats['mean']) / self.time_stats['std']
        
        obs_band_vals = band_vals[obs]
        if self.band_stats is not None:
            obs_band_vals = (obs_band_vals - self.band_stats['means']) / self.band_stats['stds']

        # return selected observations
        T, C = self.max_seq_len, len(self.bands)
        
        res_band_vals = np.zeros((T, C), dtype= np.float32)
        res_band_vals[loc] = obs_band_vals.astype(np.float32)

        res_timestamps = np.zeros(T, dtype= np.float32)
        res_timestamps[loc] = obs_timestamps.astype(np.float32)

        mask = np.zeros(T, dtype= bool)
        mask[loc] = True

        item = {
            'year_idx': y_idx,
            'parcel_id': y_parcels['ids'][p_idx],
            'label': torch.tensor(y_parcels['labels'][p_idx]),
            'n_px': torch.tensor(n_px),
            'timestamps': torch.from_numpy(res_timestamps),
            'band_vals': torch.from_numpy(res_band_vals),
            'mask': torch.from_numpy(mask)
        }

        return item


    @property
    def labels(self):
        return np.concatenate([y['p']['labels'] for y in self.y_data])

    def get_class_counts(self):
        if self.ignore_index is not None:
            return np.bincount(self.labels[self.labels != self.ignore_index], minlength= self.num_classes)
        return np.bincount(self.labels, minlength= self.num_classes)

    def get_class_weights(self, beta= 0.9):
        """Calculate class weights using effective number of samples.
        Args:
            beta: Hyperparameter for effective number calculation
        Returns:
            np.ndarray: Class weights
        """
        class_counts = self.get_class_counts()
        if self.ignore_index is not None:
            class_counts[self.ignore_index] = 0

        if beta == 1.0:
            weights = 1.0 / (class_counts + 1e-6)
        else:
            eff = 1.0 - np.power(beta, class_counts)
            eff[eff <= 0] = 1e-6
            weights = (1.0 - beta) / eff

        missing = class_counts == 0
        weights[missing] = 1e-6
        if self.ignore_index is not None:
            weights[self.ignore_index] = 0

        weights = weights / np.sum(weights) * self.num_classes

        return weights.astype(np.float32)
    
    
    def compute_band_stats(self, max_parcels= 50_000, seed= 42):
        samples = np.random.default_rng(seed).choice(len(self), size= min(max_parcels, len(self)), replace= False)

        saved = (self.training, self.band_stats)
        # set training and band_stats to sample raw values
        self.training, self.band_stats = False, None
        try:
            C = len(self.bands)
            s, ss, n_obs = np.zeros(C), np.zeros(C), 0
            for i in samples:
                item = self[int(i)]
                
                mask = item['mask'].numpy()
                if not mask.any():
                    continue
                
                band_vals = item['band_vals'].numpy()[mask].astype(np.float64)

                s += band_vals.sum(axis= 0)
                ss += (band_vals ** 2).sum(axis= 0)
                n_obs += len(band_vals)
        finally:
            self.training, self.band_stats = saved

        if n_obs == 0:
            raise RuntimeError("No valid observations found for computing band statistics.")

        means = s / n_obs
        stds = np.sqrt(np.maximum(ss / n_obs - means ** 2, 1e-8))
        
        return {'means': means.astype(np.float32), 'stds': stds.astype(np.float32), 'from_n_obs': n_obs}

    
    def compute_time_stats(self, max_timestamps= 20_000, seed= 42):
        samples = np.random.default_rng(seed).choice(len(self), size= min(max_timestamps, len(self)), replace= False)

        saved = (self.training, self.time_stats)
        # set training and time_stats to sample raw values
        self.training, self.time_stats = False, None
        try:
            s, ss, n_obs = 0, 0, 0
            for i in samples:
                item = self[int(i)]
                
                mask = item['mask'].numpy()
                if mask.any():
                    timestamps = item['timestamps'].numpy()[mask].astype(np.float64)
                    s += timestamps.sum()
                    ss += (timestamps ** 2).sum()
                    n_obs += len(timestamps)
        finally:
            self.training, self.time_stats = saved

        if n_obs == 0:
            raise RuntimeError("No valid observations found for computing time statistics.")
        
        mean = s / n_obs
        std = np.sqrt(np.maximum(ss / n_obs - mean ** 2, 1e-8))
        
        return {'mean': mean, 'std': std, 'from_n_obs': n_obs}