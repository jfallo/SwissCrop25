# SwissCrop25 Parcel Time Series

A **parcel-level time series dataset** from SwissCrop25: per-parcel Sentinel-2 observations for
Swiss agricultural parcels, 2019–2025. Built from Switzerland's LNF (Landwirtschaftliche
Nutzungsflächen) declarations on the national 10 m Sentinel-2 grid, with daily temperature and
precipitation from MeteoSwiss. Four Parquet tables, no special tooling needed beyond a Parquet
reader.

Companion dataset to the official public release,
[SwissCrop25](https://huggingface.co/datasets/EOA-team/SwissCrop25).

## Layout

```
parcels/{year}.parquet          one row per declared parcel
observations/{year}.parquet     one row per parcel per Sentinel-2 acquisition date
parcel_prev/{year}.parquet      one row per (parcel, previous-year land-use code)
climate_daily/{year}.parquet    one row per 1 km climate cell per day
```

`{year}` ranges 2019–2025 (2018–2025 for `climate_daily`). Join the first three on `parcel_id`
— unique **within** a year only (see "Parcel identity" below). `climate_daily` joins to
`parcels` on `climate_cell_e_epsg2056`, `climate_cell_n_epsg2056` (see "Daily climate" below).

## `parcels`

| column | type | meaning |
|---|---|---|
| `parcel_id` | string | unique within this year |
| `year` | int | declaration year |
| `lnf_code` | int | original LNF land-use code, unaggregated |
| `area_m2` | float | true polygon area (EPSG:32632) |
| `n_px` | int | qualifying Sentinel-2 pixels assigned to this parcel (0 if none) |
| `mean_purity` | float or null | mean land-use purity over qualifying pixels; null if `n_px` is 0 |
| `overlay_flag` | bool | true if pixels show overlapping declarations |
| `drop_reason` | string | `none`, `no_pixel_centre`, or `no_pure_pixel` |
| `tile` | string | Sentinel-2 tile holding most of this parcel's pixels |
| `centroid_e_epsg32632`, `centroid_n_epsg32632` | float | polygon centroid (EPSG:32632, metres) |
| `climate_cell_e_epsg2056`, `climate_cell_n_epsg2056` | int | centre of the 1 km MeteoSwiss cell the centroid falls in (EPSG:2056, metres) — a join key to `climate_daily`, not the centroid itself |

A parcel with `n_px = 0` still appears here (`drop_reason != 'none'`) but has no rows in
`observations` or `parcel_prev`.

## `observations`

| column | type | meaning |
|---|---|---|
| `parcel_id` | string | joins to `parcels.parcel_id`, same year |
| `date` | string (`YYYY-MM-DD`) | acquisition date |
| `doy` | int | day of year, 1–366 |
| `n_clear_px` | int | qualifying pixels judged cloud-free on this date |
| `clear_frac` | float | `n_clear_px / n_px` |
| `n_data_px` | int | qualifying pixels with any data (cloud-free or not) |
| `max_granules` | int | most MGRS granules any one pixel was reassembled from |
| `n_tiles` | int | Sentinel-2 tiles this parcel's pixels were read from |
| `s2_{B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12}_clear` | float or null | per-band median over cloud-free pixels; null if `n_clear_px` is 0 |
| `s2_{...}_all` | float | per-band median over all pixels with data (cloud not excluded) |

Every acquisition in the window is included, even fully cloudy ones — usability is
`clear_frac`/`n_clear_px`, not a boolean the export decides for you. The window runs **1
September of the previous year through 31 December** of the declaration year, so winter-sown
crops are observable before the year starts. Consecutive years therefore share four months of
acquisitions (Sep–Dec) — worth knowing if you split parcel-years into train/test.

## `parcel_prev`

| column | type | meaning |
|---|---|---|
| `parcel_id` | string | joins to `parcels.parcel_id`, same year |
| `prev_code` | string | an LNF code declared the previous year, or the literal `undeclared` |
| `fraction` | float | share of this parcel's pixels that carried `prev_code` the previous year |

For every parcel, `Σ(fraction)` across its rows is exactly 1.0 — including a rare small
**negative** `undeclared` where the *previous* year's own declarations overlapped (~7.5–7.9% of
parcels from 2021 onward, ~5.0% in 2020). Treat a negative value as "overlap detected," not
as an error.

**2019 parcels have zero `parcel_prev` rows** — LNF declarations begin in 2019, so there's no
predecessor year. That's expected, not a gap.

## Parcel identity

`parcel_id` is minted fresh each year and does **not** identify the same physical parcel across
years — a field redeclared in consecutive years gets two unrelated ids. Cross-year continuity
is carried only through `parcel_prev`'s area-share distribution.

## Daily climate

| column | type | meaning |
|---|---|---|
| `climate_cell_e_epsg2056`, `climate_cell_n_epsg2056` | int | cell centre (EPSG:2056, metres); joins to `parcels` |
| `date` | date | calendar day |
| `TabsD` | float | daily mean temperature, °C |
| `RhiresD` | float | daily precipitation sum, mm |

From the MeteoSwiss spatial climate analyses on their 1 km grid, `TabsD` v1.2 and `RhiresD`
v2.0, values exactly as MeteoSwiss provides them; see the MeteoSwiss product documentation for
their definitions. Each parcel is linked to the one cell containing its centroid, with no
interpolation or elevation adjustment. About 70 parcels share a cell, which is why the series
is stored per cell rather than per parcel.

The series runs continuously from **1 September 2018 to 31 December 2025**, one file per
calendar year (2018 holds Sep–Dec only), covering every cell any parcel falls in. A few cells
along the national border lie outside the MeteoSwiss grid's valid area and are empty throughout
(55 cells for `TabsD`, 12 for `RhiresD`; 30–120 parcels per year) — values there are null, with
no substitute from a neighbouring cell. For a
declaration year `y`, read `climate_daily/{y-1}` and `climate_daily/{y}` to cover the same
Sep–Dec window as `observations`:

```python
key = ["climate_cell_e_epsg2056", "climate_cell_n_epsg2056"]
p = pd.read_parquet("parcels/2022.parquet", columns=["parcel_id", *key])
cl = pd.concat(pd.read_parquet(f"climate_daily/{y}.parquet") for y in (2021, 2022))
series = p.merge(cl, on=key)          # parcel_id, date, TabsD, RhiresD
```

Values are raw and unaccumulated: growing-degree-days, precipitation totals or any other
derived quantity are left for you to compute with the base temperature and start date of your
choice. `observations` itself carries `doy` only.

## Known limitations

- **Pixel contamination is disclosed, not corrected.** No erosion/buffering is applied (it would
  remove roughly half a median Swiss parcel); use `mean_purity` and `n_px` to filter if you need
  cleaner pixels.
- **~14% of declared parcels carry no pixel at all** (13.6–15.6% per year), stable across all
  seven years — a property of Swiss parcel geometry against a 10 m grid. Loss concentrates in
  grassland and linear landscape elements (hedges, paths, walls); vineyards are the one arable
  class that loses meaningfully, plausibly for the same long-and-narrow-shape reason as the
  landscape elements.
- **`lnf_code` is a declaration, not a verified outcome** — it reflects a subsidy-programme
  registration, not ground-truthed crop identity or management.
- **Overlapping declarations are flagged, not resolved** (`overlay_flag`, and `parcel_prev`'s
  negative `undeclared`). Rare (well under 1% of pixels) but real.
- **No LNF code is aggregated, filtered, or remapped** — `lnf_code` is exactly as declared;
  any coarser classification is left to you.
- **Climate is at 1 km, not at the parcel.** Every parcel in a cell gets the same values; in
  alpine terrain a parcel can sit hundreds of metres above or below its cell's mean elevation.
  Parcels whose centroid falls just across the border have no climate values (see "Daily
  climate").

## Scale

```
year   parcels     with px     overlay-flagged   observations rows   parcel_prev rows
2019    916,889     789,027     26,240 (2.9%)        128,903,655                   0
2020  1,535,749   1,311,856     51,445 (3.3%)        209,472,246           2,944,742
2021  1,700,362   1,435,680     37,195 (2.2%)        213,556,367           3,951,327
2022  1,682,281   1,440,753     25,675 (1.5%)        213,770,974           4,262,289
2023  1,674,646   1,447,339     23,744 (1.4%)        225,425,334           4,289,065
2024  1,698,363   1,460,941     23,995 (1.4%)        229,420,586           4,342,354
2025  1,712,783   1,469,687     17,183 (1.0%)        259,220,874           4,376,946
```

`climate_daily`: 30,830 cells × 2,679 days (2018-09-01 to 2025-12-31), 82,593,570 rows.

## Files and integrity

`SHA256SUMS` lists a checksum for every Parquet file. After downloading, check the copy is
complete and uncorrupted from the dataset folder with:

```bash
sha256sum -c SwissCrop25-ParcelTS.tar.sha256
```

## Terms of use

This dataset is shared privately and is not yet released under any license. It must not be
redistributed, shared with third parties, or published, in whole or in part, without prior
permission from the authors.
