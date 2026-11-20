#!/usr/bin/env python3
"""
Author: Claude
Date: November 20, 2026
Description:
clusterArrays.py -- MAIA clusters as 13 x 21 pixel arrays in the DataProcessing / pixelav "recon2D" layout, plus cluster
coordinates (ylocal, zglobal) comparable to make_tracklists.py, saved as one flat table.

Input: a LinkedClusters (linkClusters.py), i.e. the tables saved by linkEdmHits.py. The pixel table is needed.

    from linkClusters import LinkedClusters
    from clusterArrays import ClusterArrayBuilder
    cl = LinkedClusters.fromFiles(stub)
    cl = cl[cl.cellFields["layer"] == 0]                 # any selection
    builder = ClusterArrayBuilder()                         # MAIA_v0 geometry, 13 x 21 arrays, flp = 0
    table = builder.buildTable(cl)                          # one row per cluster: labels + 273 pixel columns pix_0 .. pix_272
    builder.save(table, stub)                               # <stub>.vtxClusterArrays.csv / .parquet (+ DataProcessing layout)

    python clusterArrays.py <stub> --layers 0               # the same from the command line (--help)

Each step is its own method and can be used alone:
    pixelCentroids(clusters)          charge-weighted centroid in pixel indices, the centre pixel, the total charge
    arrays(clusters, centroids)       (N, 13, 21) arrays and what did not fit: n_pixels_outside, charge_outside, truncated
    coordinates(clusters, centroids)  ylocal / zglobal from the cluster position AND from the pixel indices + geometry, their
                                      difference, and a CoordinateMismatchWarning when they disagree by more than tolMm
    buildTable(clusters)              all of the above as one flat DataFrame
    save(table, stub)                 csv and/or parquet, and the DataProcessing layout (labels + recon2D parquet files)

Conventions (checked on digi_output_mu1, 417,725 clusters, all layers):
  * Pixel position.x is the pixel index across the stave (along the cluster's u axis), position.y the index along z
    (v axis). Both start at 0 at the sensor edge, the pitch is 25 um, and the orientation is the same on every stave of
    every layer. Every pixel carries its cluster's cellID.
  * Array rows run along ylocal (pixel position.x), columns along z (pixel position.y), as in DataProcessing, where a
    row shift changes y-local and a column shift changes z-global. Row index increases with ylocal. Flattened
    row-major: pixel column 'pix_k' / recon2D column 'k' is row k // 21, column k % 21.
  * The array is centred like DataProcessing's adjustCluster: the charge-weighted centroid, rounded to the nearest pixel
    (numpy rounding, halves to even), sits at row 6, column 10. Differences to DataProcessing: the centroid uses all
    pixels of the cluster (also those that do not fit in the array); DataProcessing's row sums take 22 values per row
    (lastSlice[i*21:(i+1)*21+1]), this code the 21 of each row; there is one time slice (the pixel 'time' is 0 here).
  * Truncation: pixels that do not fit in the 13 x 21 window are left out and counted. The row is kept and flagged:
    truncated = True, with n_pixels_outside and charge_outside; array_empty = True if not one pixel is inside (a sprawling
    cluster whose centroid lies where no pixel fired). Nothing is dropped. On digi_output_mu1: 2.8% / 3.3% truncated in
    layers 0 / 1, below 0.6% in the others, 8 empty. For truncated clusters this centring (centroid of the WHOLE cluster)
    differs from DataProcessing, which only sees the window; for untruncated ones its adjustCluster shifts nothing.
  * ylocal / zglobal follow make_tracklists.py: y_rot is the position across the stave after rotating (x, y) by the
    stave angle gamma, ylocal = +y_rot for flp = 0 and -y_rot for flp = 1, zglobal = global z, gamma in [0, 2 pi).
    make_tracklists rounds both to the 25 um pitch; here they are kept unrounded (about 1% of clusters sit exactly on a
    pixel boundary, where rounding is decided by float noise). make_tracklists' getYlocalAndGamma only works for layer 0
    (it hard-codes the layer-0 stave distance 30.165 mm and 16 staves); here gamma comes from each cluster's own u axis,
    which gives the same gamma as getYlocalAndGamma for layer 0 (5000 of 5000 tested) and the right one for all layers.
  * From the pixel indices: y_rot = -width/2 + offset + (x + 0.5) * pitch and z = -zmax + sensor * length + (y + 0.5) *
    pitch, with the geometry below. In z this matches the cluster position (median difference 0.0000 mm); across the
    stave it differs by about -0.019 mm in every layer. That is the Lorentz drift: the digitiser (MuonCVXDRealDigitiser,
    TanLorentz = 0.8 by default, 0 along z) shifts the charge by tan * distance to the readout plane, about 0.8 * 25 um,
    and corrects the cluster position for it. So *_pos is the best estimate of where the particle crossed (on
    digi_output_H1 the 17 signal sim hits lie within 0.008 mm of ylocal_pos), while *_pix and *_center describe where
    the charge landed (about 0.02 mm away). The warning reports the per-layer offset so you can see it.
  * Charge: the pixel eDep is the charge in electrons (the digitiser stores it so); charge_total equals the cluster eDep
    times ElectronsPerKeV (270.3 e/keV by default). Pixel 'time' is not a measured hit time but the END of the
    digitiser's 25 ns time window: 0 for sim hits in [-25, 0) ns, 25 for [0, 25) ns (checked on digi_output_H1 against the
    linked sim hit times). A cluster is built inside one window, so all its pixels share it (all 398,975 clusters of H1)
    and one 2D array per cluster is complete; there are no finer time slices like DataProcessing's recon3D.
"""
import argparse
import os
import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd


class CoordinateMismatchWarning(UserWarning):
    """ylocal / zglobal from the cluster position and from the pixel indices disagree.
    warnings.simplefilter("error", CoordinateMismatchWarning) turns it into an error."""


@dataclass(frozen=True)
class BarrelGeometry:
    """Vertex barrel geometry. Defaults: MAIA_v0, from key4hep/k4geo MuColl/MAIA/compact/MAIA_v0 (commit 16658a9),
    Vertex_o2_v06_01.xml. Layers are the cellID 'layer' values (0 and 1 are the double layer)."""
    pitchMm: float = 0.025
    zmaxMm: float = 65.0                      # VertexBarrel_zmax
    nSensorsZ: int = 5                        # VertexBarrel_nmodules: sensors along z = the cellID 'sensor' field
    staveOffsetMm: float = 2.0                # VertexBarrel_LayerN_offset (the same for every layer)
    widthMm: tuple = ((0, 13.0), (1, 13.0), (2, 23.0), (4, 24.0), (6, 24.0))     # (cellID layer, stave width)

    @property
    def sensorLengthMm(self):
        return 2 * self.zmaxMm / self.nSensorsZ

    def width(self, layer):
        """stave width for each layer value; NaN for a layer that is not in widthMm"""
        table = dict(self.widthMm)
        return np.array([table.get(int(l), np.nan) for l in np.atleast_1d(layer)], dtype=float)

    def yrotFromPixel(self, layer, x):
        """position across the stave (rotated frame, as in make_tracklists' yentry) of pixel index x (may be fractional)"""
        return -self.width(layer) / 2 + self.staveOffsetMm + (np.asarray(x, dtype=float) + 0.5) * self.pitchMm

    def zFromPixel(self, sensor, y):
        """global z of pixel index y (may be fractional) on sensor `sensor`"""
        return -self.zmaxMm + np.asarray(sensor, dtype=float) * self.sensorLengthMm + (np.asarray(y, dtype=float) + 0.5) * self.pitchMm


@dataclass(frozen=True)
class ArrayConfig:
    """The cluster arrays. Defaults: the DataProcessing / pixelav layout (13 rows along ylocal, 21 columns along z)."""
    nRows: int = 13
    nCols: int = 21
    chargeColumn: str = "eDep"                # pixel column used as the charge (electrons for the MAIA pixels)
    flp: int = 0                              # as make_tracklists --flp: 0 -> ylocal = +y_rot, 1 -> ylocal = -y_rot
    dtype: str = "float32"

    @property
    def centerRow(self):
        return self.nRows // 2

    @property
    def centerCol(self):
        return self.nCols // 2


class ClusterArrayBuilder:
    # ---- class constants --------------------------------------------------------------------------------------------
    DEFAULT_NAME = "vtxClusterArrays"                     # saved as <stub>.<name>.csv / .parquet / .labels.parquet / .recon2D.parquet
    PIXEL_X, PIXEL_Y = "position.x", "position.y"         # pixel index across the stave / along z
    CLUSTER_POSITION = ("position.x", "position.y", "position.z")
    CLUSTER_U = ("u.a", "u.b")                            # theta, phi of the cluster's u axis
    DEFAULT_EXTRA_CLUSTER_COLUMNS = ("eDep", "time", "position.x", "position.y", "position.z")   # copied as cluster_<name>
    DEFAULT_TOL_MM = 0.05                                 # two pixels; on digi_output_mu1 0.6% of clusters exceed it
    PIXEL_PREFIX = "pix_"

    def __init__(self, geometry=None, config=None, tolMm=None, extraClusterColumns=None, nExamples=5):
        self.geometry = geometry if geometry is not None else BarrelGeometry()
        self.config = config if config is not None else ArrayConfig()
        self.tolMm = self.DEFAULT_TOL_MM if tolMm is None else tolMm
        self.extraClusterColumns = tuple(self.DEFAULT_EXTRA_CLUSTER_COLUMNS if extraClusterColumns is None else extraClusterColumns)
        self.nExamples = nExamples
        if self.config.flp not in (0, 1):
            raise ValueError("flp must be 0 or 1")

    # ---- helpers --------------------------------------------------------------------------------------------------------
    @staticmethod
    def _owner(counts):
        return np.repeat(np.arange(len(counts)), counts)

    @staticmethod
    def _asIndex(values, name):
        idx = np.rint(values).astype(np.int64)
        if len(values) and not np.allclose(idx, values):
            raise ValueError(f"pixel {name} are not whole pixel indices")
        return idx

    @property
    def ylocalSign(self):
        """ylocal = sign * y_rot (make_tracklists: flp 0 flips its -y_rot back to +y_rot)"""
        return 1.0 if self.config.flp == 0 else -1.0

    @classmethod
    def staveAngle(cls, table):
        """gamma of each cluster's stave: the angle of the outward normal of its u axis, in [0, 2 pi) like make_tracklists"""
        a, b = (table[c].to_numpy(dtype=float) for c in cls.CLUSTER_U)
        ux, uy = np.sin(a) * np.cos(b), np.sin(a) * np.sin(b)
        nx, ny = uy, -ux
        x, y = (table[c].to_numpy(dtype=float) for c in cls.CLUSTER_POSITION[:2])
        inward = nx * x + ny * y < 0
        nx, ny = np.where(inward, -nx, nx), np.where(inward, -ny, ny)
        return np.mod(np.arctan2(ny, nx), 2 * np.pi)

    # ---- step 1: centroids ----------------------------------------------------------------------------------------------
    def pixelCentroids(self, clusters):
        """Per cluster: charge-weighted centroid of the pixel indices (centroid_px, centroid_py, as stored in pixel
        position.x / position.y), the centre pixel (center_px, center_py: the centroid rounded, halves to even, as in
        DataProcessing; -1 without pixels), the total charge and the number of pixels."""
        x = clusters.pixelColumn(self.PIXEL_X)
        y = clusters.pixelColumn(self.PIXEL_Y).flat
        q = clusters.pixelColumn(self.config.chargeColumn).flat.astype(float)
        counts, n = x.counts, len(clusters)
        owner = self._owner(counts)
        xs = x.flat.astype(float)
        Q = np.bincount(owner, weights=q, minlength=n)
        # clusters whose charge sums to 0 fall back to the plain mean, so they still get a position
        w = np.where((Q > 0)[owner], q, 1.0)
        W = np.bincount(owner, weights=w, minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            cx = np.bincount(owner, weights=w * xs, minlength=n) / W
            cy = np.bincount(owner, weights=w * y.astype(float), minlength=n) / W
        has = counts > 0
        return pd.DataFrame({"centroid_px": np.where(has, cx, np.nan), "centroid_py": np.where(has, cy, np.nan),
                             "center_px": np.where(has, np.rint(np.nan_to_num(cx)), -1).astype(np.int64),
                             "center_py": np.where(has, np.rint(np.nan_to_num(cy)), -1).astype(np.int64),
                             "charge_total": Q, "n_pixels": counts.astype(np.int64)})

    # ---- step 2: arrays -------------------------------------------------------------------------------------------------
    def arrays(self, clusters, centroids=None):
        """(N, nRows, nCols) arrays centred on the centre pixel, and a DataFrame with charge_in_array, n_pixels_outside,
        charge_outside, truncated (pixels that did not fit are left out of the array, the cluster is kept) and
        array_empty (truncated with nothing at all inside the window)."""
        cfg = self.config
        cent = self.pixelCentroids(clusters) if centroids is None else centroids
        xr = clusters.pixelColumn(self.PIXEL_X)
        x = self._asIndex(xr.flat, "position.x")
        y = self._asIndex(clusters.pixelColumn(self.PIXEL_Y).flat, "position.y")
        q = clusters.pixelColumn(cfg.chargeColumn).flat.astype(float)
        n, owner = len(clusters), self._owner(xr.counts)
        rowSign = 1 if cfg.flp == 0 else -1                      # rows increase with ylocal
        row = rowSign * (x - cent["center_px"].to_numpy()[owner]) + cfg.centerRow
        col = (y - cent["center_py"].to_numpy()[owner]) + cfg.centerCol
        inside = (row >= 0) & (row < cfg.nRows) & (col >= 0) & (col < cfg.nCols)
        flat = np.zeros((n, cfg.nRows * cfg.nCols), dtype=cfg.dtype)
        np.add.at(flat, (owner[inside], (row * cfg.nCols + col)[inside]), q[inside])
        out = ~inside
        info = pd.DataFrame({"charge_in_array": np.bincount(owner[inside], weights=q[inside], minlength=n),
                             "n_pixels_outside": np.bincount(owner[out], minlength=n).astype(np.int64),
                             "charge_outside": np.bincount(owner[out], weights=q[out], minlength=n)})
        info["truncated"] = info["n_pixels_outside"].to_numpy() > 0
        # a sprawling cluster can have its charge-weighted centroid where no pixel fired: then nothing is in the window
        info["array_empty"] = info["truncated"].to_numpy() & (np.bincount(owner[inside], minlength=n) == 0)
        return flat.reshape(n, cfg.nRows, cfg.nCols), info

    # ---- step 3: coordinates ---------------------------------------------------------------------------------------------
    def coordinates(self, clusters, centroids=None, warn=True):
        """ylocal / zglobal of each cluster two ways, in mm:
            *_pos     from the cluster's global position (make_tracklists convention, unrounded)
            *_pix     from the charge-weighted centroid of the pixel indices and the geometry
            *_center  of the centre pixel the array is centred on (pixel aligned, like DataProcessing's labels)
        d_ylocal / d_zglobal = pos - pix; coords_disagree where |d| > tolMm; coords_unavailable where the pixel way
        cannot be computed (no pixels, or a layer not in the geometry). Warns (CoordinateMismatchWarning) if either."""
        cent = self.pixelCentroids(clusters) if centroids is None else centroids
        tab = clusters.clusterTable
        own = clusters.cellFields
        layer, sensor = own["layer"], own["sensor"]
        gamma = self.staveAngle(tab)
        x, y, z = (tab[c].to_numpy(dtype=float) for c in self.CLUSTER_POSITION)
        sgn, geo = self.ylocalSign, self.geometry
        yrot_pos = y * np.cos(gamma) - x * np.sin(gamma)
        out = pd.DataFrame({"gamma": gamma,
                            "ylocal_pos": sgn * yrot_pos, "zglobal_pos": z,
                            "ylocal_pix": sgn * geo.yrotFromPixel(layer, cent["centroid_px"].to_numpy()),
                            "zglobal_pix": geo.zFromPixel(sensor, cent["centroid_py"].to_numpy())})
        has = cent["n_pixels"].to_numpy() > 0
        out["ylocal_center"] = np.where(has, sgn * geo.yrotFromPixel(layer, cent["center_px"].to_numpy()), np.nan)
        out["zglobal_center"] = np.where(has, geo.zFromPixel(sensor, cent["center_py"].to_numpy()), np.nan)
        out["d_ylocal"] = out["ylocal_pos"] - out["ylocal_pix"]
        out["d_zglobal"] = out["zglobal_pos"] - out["zglobal_pix"]
        dy, dz = out["d_ylocal"].to_numpy(), out["d_zglobal"].to_numpy()
        unavailable = ~np.isfinite(dy) | ~np.isfinite(dz)
        out["coords_unavailable"] = unavailable
        out["coords_disagree"] = ~unavailable & ((np.abs(dy) > self.tolMm) | (np.abs(dz) > self.tolMm))
        if warn:
            self._warnCoordinates(out, layer, cent["n_pixels"].to_numpy(), clusters.rows)
        return out

    def _warnCoordinates(self, coords, layer, nPixels, rows):
        bad, unav = coords["coords_disagree"].to_numpy(), coords["coords_unavailable"].to_numpy()
        if not (bad.any() or unav.any()):
            return
        lines = [f"ylocal/zglobal from the cluster position and from the pixel indices differ by more than {self.tolMm} mm "
                 f"for {int(bad.sum()):,} of {len(coords):,} clusters ({100 * bad.mean():.2f}%); unavailable for {int(unav.sum()):,}."]
        ok = ~unav
        if ok.any():
            per = pd.DataFrame({"layer": layer[ok], "d_ylocal": coords["d_ylocal"].to_numpy()[ok],
                                "d_zglobal": coords["d_zglobal"].to_numpy()[ok]}).groupby("layer").median()
            lines.append("median (pos - pix) per layer [mm]: " + ", ".join(
                f"layer {l}: dy {r.d_ylocal:+.4f} dz {r.d_zglobal:+.4f}" for l, r in per.iterrows()))
        if bad.any():
            ex = pd.DataFrame({"cluster_row": rows[bad], "layer": layer[bad], "n_pixels": nPixels[bad],
                               "d_ylocal": coords["d_ylocal"].to_numpy()[bad], "d_zglobal": coords["d_zglobal"].to_numpy()[bad]})
            ex = ex.reindex(ex[["d_ylocal", "d_zglobal"]].abs().max(axis=1).sort_values(ascending=False).index).head(self.nExamples)
            lines.append("largest differences:\n" + ex.round(4).to_string(index=False))
        warnings.warn("\n".join(lines), CoordinateMismatchWarning, stacklevel=3)

    # ---- step 4: the flat table ------------------------------------------------------------------------------------------
    def buildTable(self, clusters, includeArrays=True, warn=True):
        """One row per cluster: identifiers, counts, the cluster's own cellID fields, centroid, coordinates (both ways),
        truncation, copies of some cluster columns (cluster_<name>), and the array as pix_0 .. pix_{nRows*nCols-1}."""
        tab = clusters.clusterTable
        own = clusters.cellFields
        cent = self.pixelCentroids(clusters)
        table = pd.DataFrame({"ievent": tab["ievent"].to_numpy(), "cluster_index": tab["index"].to_numpy(),
                              "cluster_row": clusters.rows})
        for k in ("layer", "module", "sensor"):
            if k in own:
                table[k] = own[k]
        table["n_sim_hits"] = clusters.nSimHits
        try:
            table["n_signal_sim_hits"] = clusters.nSignal
        except RuntimeError:
            pass                                       # the sim hit table has no overlay column
        table = pd.concat([table, cent], axis=1)
        table = pd.concat([table, self.coordinates(clusters, cent, warn=warn)], axis=1)
        arr, info = self.arrays(clusters, cent)
        table = pd.concat([table, info], axis=1)
        for c in self.extraClusterColumns:
            if c in tab:
                table[f"cluster_{c}"] = tab[c].to_numpy()
        if includeArrays:
            k = self.config.nRows * self.config.nCols
            pix = pd.DataFrame(arr.reshape(len(arr), k), columns=[f"{self.PIXEL_PREFIX}{i}" for i in range(k)])
            table = pd.concat([table, pix], axis=1)
        return table

    # ---- step 5: saving --------------------------------------------------------------------------------------------------
    def save(self, table, fileStub, name=None, csv=True, parquet=True, dataProcessingLayout=True):
        """<stub>.<name>.csv and/or .parquet (the flat table); with dataProcessingLayout also <stub>.<name>.labels.parquet
        (every non-pixel column) and <stub>.<name>.recon2D.parquet (the array, columns '0'..'272', like DataProcessing)."""
        name = name or self.DEFAULT_NAME
        base = f"{fileStub}.{name}"
        written = []
        if csv:
            table.to_csv(f"{base}.csv", index=False); written.append(f"{base}.csv")
        if parquet:
            table.to_parquet(f"{base}.parquet", index=False); written.append(f"{base}.parquet")
        pixCols = [c for c in table.columns if c.startswith(self.PIXEL_PREFIX)]
        if dataProcessingLayout and pixCols:
            table.drop(columns=pixCols).to_parquet(f"{base}.labels.parquet", index=False)
            recon = table[pixCols].copy()
            recon.columns = [c[len(self.PIXEL_PREFIX):] for c in pixCols]
            recon.to_parquet(f"{base}.recon2D.parquet", index=False)
            written += [f"{base}.labels.parquet", f"{base}.recon2D.parquet"]
        for w in written:
            print(f"saved {len(table):,} clusters -> {w} ({os.path.getsize(w) / 1e6:.1f} MB)")
        return written


# ----------------------------------------------------------------------------------------------------------------------
def selectClusters(clusters, layers=None, signalOnly=False):
    """a selection of a LinkedClusters by layer (cluster's own cellID) and/or clusters with a signal sim hit"""
    keep = np.ones(len(clusters), bool)
    if layers is not None:
        keep &= np.isin(clusters.cellFields["layer"], list(layers))
    if signalOnly:
        keep &= clusters.nSignal > 0
    return clusters[keep]


def parseArgs(argv=None):
    a, g = ArrayConfig(), ClusterArrayBuilder
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fileStub", help="path of the tables saved by linkEdmHits.py, without '.<table>.parquet'")
    ap.add_argument("--layers", type=int, nargs="*", default=None, help="only these cellID layers (default: all)")
    ap.add_argument("--signal-only", action="store_true", help="only clusters with a signal sim hit")
    ap.add_argument("--flp", type=int, default=a.flp, choices=(0, 1), help="as make_tracklists (default: %(default)s)")
    ap.add_argument("--tol-mm", type=float, default=g.DEFAULT_TOL_MM, help="coordinate mismatch warning above this (default: %(default)s)")
    ap.add_argument("--name", default=g.DEFAULT_NAME, help="output name (default: %(default)s)")
    ap.add_argument("--no-csv", action="store_true"); ap.add_argument("--no-parquet", action="store_true")
    ap.add_argument("--no-dataprocessing-layout", action="store_true", help="do not write the labels / recon2D parquet files")
    return ap.parse_args(argv)


def main(argv=None):
    a = parseArgs(argv)
    from linkClusters import LinkedClusters
    clusters = selectClusters(LinkedClusters.fromFiles(a.fileStub), layers=a.layers, signalOnly=a.signal_only)
    print(f"{clusters} from {a.fileStub}")
    builder = ClusterArrayBuilder(config=ArrayConfig(flp=a.flp), tolMm=a.tol_mm)
    table = builder.buildTable(clusters)
    print(f"truncated clusters (pixels outside {builder.config.nRows} x {builder.config.nCols}): {int(table['truncated'].sum()):,} of {len(table):,}")
    builder.save(table, a.fileStub, name=a.name, csv=not a.no_csv, parquet=not a.no_parquet,
                 dataProcessingLayout=not a.no_dataprocessing_layout)
    return 0


if __name__ == "__main__":
    sys.exit(main())