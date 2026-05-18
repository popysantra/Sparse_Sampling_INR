"""
sampling.py - Different sampling strategies for sparse observations
Supports: random, uniform grid, stratified, importance, edge-biased, 
          random-stratified, and latin hypercube sampling
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
from scipy.spatial import cKDTree
from scipy.stats import qmc
import scipy as sp
from math import floor, ceil, acos
from skimage.segmentation import slic
from skimage.util import img_as_float
import torch


class SamplingType(Enum):
    """Available sampling strategies."""
    RANDOM = "random"
    UNIFORM_GRID = "uniform_grid"
    STRATIFIED = "stratified"
    IMPORTANCE = "importance"
    EDGE_BIASED = "edge_biased"
    RANDOM_STRATIFIED = "random_stratified"
    LATIN_HYPERCUBE = "latin_hypercube"
    GRID_AGGREGATE = "grid_aggregate"  
    HISTOGRAM = "histogram"               # value histogram only
    GRADIENT_HISTOGRAM = "gradient_histogram"   # gradient magnitude histogram
    HISTOGRAM_GRADIENT = "histogram_gradient"   # 2D (value, gradient)
    SLIC = "slic"

    
@dataclass
class SamplingConfig:
    """Configuration for sampling strategies."""
    sampling_type: SamplingType
    n_samples: int
    n_bins: int = 10          # For histogram-based sampling
    n_strata: int = 32          # For stratified sampling
    edge_weight: float = 2.0    # For edge-biased sampling
    random_seed: int = 42
    importance_metric: str = "gradient"   # 'gradient' or 'variance'
    grid_aggregate_cells: int = 10        # Grid cells per dimension for GA sampling
    histogram_bins: int = 10              # Number of bins for histogram-based sampling

class BaseSamplingStrategy:
    """Base class for sampling strategies."""
    
    def __init__(self, config: SamplingConfig):
        self.config = config
        self.rng = np.random.RandomState(config.random_seed)
    
    def sample(self, coordinates: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample points from the dataset.
        
        Args:
            coordinates: Array of shape (N, d) where d is spatial dimension
            values: Array of shape (N,) field values
        
        Returns:
            sampled_coords: Sampled coordinates (n_samples, d)
            sampled_values: Sampled values (n_samples,)
            sampled_indices: Indices of sampled points (n_samples,)
        """
        raise NotImplementedError


class RandomSampling(BaseSamplingStrategy):
    """Simple random sampling."""
    
    def sample(self, coordinates: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_points = int(len(coordinates))
        rng = np.random.default_rng(self.config.random_seed)
        random_indices = rng.choice(n_points, size=self.config.n_samples, replace=False)
        return coordinates[random_indices], values[random_indices]
   
    
class UniformGridSampling(BaseSamplingStrategy):
    """Uniform grid sampling (regular spacing in coordinate space)."""
    
    def sample(self, coordinates: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_points = int(len(coordinates))
        
        # Estimate grid size based on desired samples
        if coordinates.shape[1] == 2:
            grid_size = int(np.ceil(np.sqrt(self.config.n_samples)))
        else:
            grid_size = int(np.ceil(self.config.n_samples ** (1/3)))
        
        selected_indices = []
        
        if coordinates.shape[1] == 2:
            # 2D case
            x_coords = coordinates[:, 0]
            y_coords = coordinates[:, 1]
            
            x_bins = np.linspace(x_coords.min(), x_coords.max(), grid_size + 1)
            y_bins = np.linspace(y_coords.min(), y_coords.max(), grid_size + 1)
            
            for i in range(grid_size):
                for j in range(grid_size):
                    mask = (x_coords >= x_bins[i]) & (x_coords < x_bins[i+1]) & \
                           (y_coords >= y_bins[j]) & (y_coords < y_bins[j+1])
                    candidates = np.where(mask)[0]
                    if len(candidates) > 0:
                        selected_indices.append(self.rng.choice(candidates))
                    if len(selected_indices) >= self.config.n_samples:
                        break
                if len(selected_indices) >= self.config.n_samples:
                    break
        else:
            # 3D case
            x_coords = coordinates[:, 0]
            y_coords = coordinates[:, 1]
            z_coords = coordinates[:, 2]
            
            x_bins = np.linspace(x_coords.min(), x_coords.max(), grid_size + 1)
            y_bins = np.linspace(y_coords.min(), y_coords.max(), grid_size + 1)
            z_bins = np.linspace(z_coords.min(), z_coords.max(), grid_size + 1)
            
            for i in range(grid_size):
                for j in range(grid_size):
                    for k in range(grid_size):
                        mask = (x_coords >= x_bins[i]) & (x_coords < x_bins[i+1]) & \
                               (y_coords >= y_bins[j]) & (y_coords < y_bins[j+1]) & \
                               (z_coords >= z_bins[k]) & (z_coords < z_bins[k+1])
                        candidates = np.where(mask)[0]
                        if len(candidates) > 0:
                            selected_indices.append(self.rng.choice(candidates))
                        if len(selected_indices) >= self.config.n_samples:
                            break
                    if len(selected_indices) >= self.config.n_samples:
                        break
                if len(selected_indices) >= self.config.n_samples:
                    break
        
        selected_indices = np.array(selected_indices[:self.config.n_samples])
        return coordinates[selected_indices], values[selected_indices]


class StratifiedSampling(BaseSamplingStrategy):
    """Stratified sampling based on value ranges."""
    
    def sample(self, coordinates: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Create strata based on value quantiles
        percentiles = np.linspace(0, 100, self.config.n_strata + 1)
        strata_boundaries = np.percentile(values, percentiles[1:-1])
        
        selected_indices = []
        samples_per_stratum = max(1, self.config.n_samples // self.config.n_strata)
        
        for i in range(self.config.n_strata):
            if i == 0:
                mask = values <= strata_boundaries[i]
            elif i == self.config.n_strata - 1:
                mask = values > strata_boundaries[-1]
            else:
                mask = (values > strata_boundaries[i-1]) & (values <= strata_boundaries[i])
            
            stratum_indices = np.where(mask)[0]
            
            if len(stratum_indices) > 0:
                n_sample = min(samples_per_stratum, len(stratum_indices))
                sampled = self.rng.choice(stratum_indices, n_sample, replace=False)
                selected_indices.extend(sampled)
        
        # If we need more samples, add random ones
        if len(selected_indices) < self.config.n_samples:
            remaining = self.config.n_samples - len(selected_indices)
            remaining_indices = [i for i in range(len(coordinates)) if i not in selected_indices]
            if len(remaining_indices) > 0:
                additional = self.rng.choice(remaining_indices, min(remaining, len(remaining_indices)), replace=False)
                selected_indices.extend(additional)
        
        selected_indices = np.array(selected_indices[:self.config.n_samples])
        return coordinates[selected_indices], values[selected_indices]


class ImportanceSampling(BaseSamplingStrategy):
    """Importance sampling based on local variance or gradient magnitude."""
    
    def _compute_local_variance(self, coordinates: np.ndarray, values: np.ndarray, k: int = 10) -> np.ndarray:
        """Compute local variance as importance score."""
        tree = cKDTree(coordinates)
        importance = np.zeros(len(values))
        
        for i in range(len(values)):
            distances, indices = tree.query(coordinates[i], k=min(k, len(values)))
            if len(indices) > 1:
                importance[i] = np.std(values[indices])
        
        return importance
    
    def _compute_gradient_magnitude(self, coordinates: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Estimate gradient magnitude using finite differences."""
        # For regular grids, we can compute gradients efficiently
        # For irregular grids, use local linear regression
        tree = cKDTree(coordinates)
        importance = np.zeros(len(values))
        
        for i in range(len(values)):
            distances, indices = tree.query(coordinates[i], k=min(6, len(values)))
            if len(indices) >= 3:
                # Fit local linear model
                local_coords = coordinates[indices]
                local_vals = values[indices]
                
                # Center the coordinates
                center = local_coords.mean(axis=0)
                local_coords_centered = local_coords - center
                
                # Solve for gradient
                try:
                    coeffs = np.linalg.lstsq(local_coords_centered, local_vals - local_vals.mean(), rcond=None)[0]
                    importance[i] = np.linalg.norm(coeffs)
                except:
                    importance[i] = 0
        
        return importance
    
    def sample(self, coordinates: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.config.importance_metric == "variance":
            importance = self._compute_local_variance(coordinates, values)
        else:  # gradient
            importance = self._compute_gradient_magnitude(coordinates, values)
        
        # Add small constant to avoid zero probabilities
        importance = importance + 1e-6
        probabilities = importance / importance.sum()
        
        indices = self.rng.choice(len(coordinates), self.config.n_samples, replace=False, p=probabilities)
        
        return coordinates[indices], values[indices]


class EdgeBiasedSampling(BaseSamplingStrategy):
    """Sample more points near domain boundaries."""
    
    def sample(self, coordinates: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Compute distance to boundary
        boundaries = []
        for i in range(coordinates.shape[1]):
            boundaries.append((coordinates[:, i].min(), coordinates[:, i].max()))
        
        dist_to_boundary = np.ones(len(coordinates)) * np.inf
        
        for i, (min_val, max_val) in enumerate(boundaries):
            dist_to_min = coordinates[:, i] - min_val
            dist_to_max = max_val - coordinates[:, i]
            dist_to_boundary_i = np.minimum(dist_to_min, dist_to_max)
            dist_to_boundary = np.minimum(dist_to_boundary, dist_to_boundary_i)
        
        # Inverse distance weighting (closer to boundary = higher weight)
        weights = 1.0 / (dist_to_boundary + 1e-6) ** self.config.edge_weight
        
        probabilities = weights / weights.sum()
        indices = self.rng.choice(len(coordinates), self.config.n_samples, replace=False, p=probabilities)
        
        return coordinates[indices], values[indices]

class RandomSamplingStratified(BaseSamplingStrategy):
    """Simple random sampling."""
    
    def sample(self, coordinates: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_points = int(len(coordinates))
        rng = np.random.default_rng(self.config.random_seed)
        random_indices = rng.choice(n_points, size=self.config.n_samples, replace=False)
        return coordinates[random_indices], values[random_indices], random_indices



class RandomStratifiedSampling(BaseSamplingStrategy):
    """Combination of random and stratified sampling."""
    
    def sample(self, coordinates: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_random = self.config.n_samples // 2
        n_stratified = self.config.n_samples - n_random
        
        # Random sampling
        random_sampler = RandomSamplingStratified(SamplingConfig(
            sampling_type=SamplingType.RANDOM,
            n_samples=n_random,
            random_seed=self.config.random_seed
        ))
        random_coords, random_vals, random_indices = random_sampler.sample(coordinates, values)
        
        # Stratified sampling on remaining points
        remaining_mask = np.ones(len(coordinates), dtype=bool)
        remaining_mask[random_indices] = False
        remaining_coords = coordinates[remaining_mask]
        remaining_vals = values[remaining_mask]
        
        stratified_sampler = StratifiedSampling(SamplingConfig(
            sampling_type=SamplingType.STRATIFIED,
            n_samples=n_stratified,
            n_strata=self.config.n_strata,
            random_seed=self.config.random_seed + 1
        ))
        strat_coords, strat_vals, strat_relative_indices = stratified_sampler.sample(remaining_coords, remaining_vals)
        
        # Convert relative indices to absolute
        strat_abs_indices = np.where(remaining_mask)[0][strat_relative_indices]
        
        # Combine
        all_coords = np.vstack([random_coords, strat_coords])
        all_vals = np.concatenate([random_vals, strat_vals])
        all_indices = np.concatenate([random_indices, strat_abs_indices])
        
        return all_coords, all_vals


class LatinHypercubeSampling(BaseSamplingStrategy):
    """Latin Hypercube sampling for better space coverage."""
    
    def sample(self, coordinates: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_dim = coordinates.shape[1]
        
        # Create Latin Hypercube samples in normalized space [0, 1]^d
        sampler = qmc.LatinHypercube(d=n_dim, seed=self.config.random_seed)
        samples = sampler.random(n=self.config.n_samples)
        
        # Scale to coordinate ranges
        scaled_samples = np.zeros_like(samples)
        for i in range(n_dim):
            min_val = coordinates[:, i].min()
            max_val = coordinates[:, i].max()
            scaled_samples[:, i] = samples[:, i] * (max_val - min_val) + min_val
        
        # Find nearest neighbors to the sampled points
        tree = cKDTree(coordinates)
        distances, indices = tree.query(scaled_samples)
        
        return coordinates[indices], values[indices]


class GridAggregateSampling(BaseSamplingStrategy):
    """
    Optimised Grid Aggregation (GA) coreset sampling.

    The coordinate space is partitioned into a regular grid of cells
    (n_cells_per_dim per spatial dimension).  One representative point is
    selected per non-empty cell by averaging all points inside it – both
    spatially and in value.  This gives a spatially-uniform, noise-reduced
    coreset that is much more informative than pure random sampling.

    The number of cells per dimension is derived from `n_samples`:
        n_cells_per_dim = ceil(n_samples ** (1/d))
    so the actual coreset size may differ slightly from `n_samples`.

    Parameters
    ----------
    config.n_samples            : target coreset size (controls grid resolution)
    config.grid_aggregate_cells : override cells-per-dim (0 = auto from n_samples)
    """

    def _build_grid(self, coordinates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (cell_indices, min_vals, gridsize) for the coordinate array."""
        n_dim = coordinates.shape[1]
        n_cells = self.config.grid_aggregate_cells
        if n_cells <= 0:
            # Auto: derive from desired sample count
            n_cells = max(2, int(np.ceil(self.config.n_samples ** (1.0 / n_dim))))

        min_vals = coordinates.min(axis=0)
        max_vals = coordinates.max(axis=0)
        gridsize  = (max_vals - min_vals) / n_cells
        # Avoid division by zero for degenerate axes
        gridsize[gridsize == 0] = 1.0

        cell_indices = np.floor((coordinates - min_vals) / gridsize).astype(int)
        # Clip to [0, n_cells-1] so boundary points stay in the last cell
        cell_indices = np.clip(cell_indices, 0, n_cells - 1)
        return cell_indices, min_vals, gridsize

    def sample(
        self, coordinates: np.ndarray, values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns
        -------
        coreset_coords  : (M, d)   centroid coordinates per non-empty cell
        coreset_values  : (M,)     mean scalar value per non-empty cell
        representative_indices : (M,) index of the data point closest to each
                                       cell centroid (needed for index-based APIs)
        """
        cell_indices, min_vals, gridsize = self._build_grid(coordinates)

        # Group point indices by cell key (tuple of ints)
        cell_map: dict = {}
        for pt_idx, cell_key in enumerate(map(tuple, cell_indices)):
            cell_map.setdefault(cell_key, []).append(pt_idx)

        coreset_coords  = []
        coreset_values  = []
        repr_indices    = []

        for cell_key, pt_indices in cell_map.items():
            pts  = coordinates[pt_indices]          # (k, d)
            vals = values[pt_indices]               # (k,)

            centroid       = pts.mean(axis=0)       # spatial mean
            centroid_val   = vals.mean()            # value mean

            # Pick the point closest to the centroid as the representative index
            dists  = np.linalg.norm(pts - centroid, axis=1)
            repr_i = pt_indices[int(np.argmin(dists))]

            coreset_coords.append(centroid)
            coreset_values.append(centroid_val)
            repr_indices.append(repr_i)

        coreset_coords = np.array(coreset_coords, dtype=np.float32)   # (M, d)
        coreset_values = np.array(coreset_values, dtype=np.float32)   # (M,)
        repr_indices   = np.array(repr_indices,   dtype=np.int64)     # (M,)

        actual = len(coreset_coords)
        target = self.config.n_samples
        print(f"  [GA Sampling] target={target}, actual coreset size={actual} "
              f"({100*actual/max(len(coordinates),1):.2f}% of input)")

        return coreset_coords, coreset_values

class HistogramSampling(BaseSamplingStrategy):
    """
    Value histogram‑based sampling.

    Steps:
      1. Build histogram of scalar values.
      2. Compute desired number of samples per bin: first each bin gets a base
         (total_samples / nbins), then remaining samples are distributed to bins
         with the largest counts (original script).
      3. Compute acceptance probability per bin = desired_count / bin_count.
      4. For each point, probability = acceptance_prob[bin_idx].
      5. Sample points with those probabilities.
    """
    def sample(self, coordinates: np.ndarray, values: np.ndarray):
        n_points = len(values)
        nbins = self.config.n_strata   # reuse n_strata as number of bins
        target_samples = self.config.n_samples

        # 1. Histogram
        counts, bin_edges = np.histogram(values, bins=nbins)
        bin_idx = np.digitize(values, bin_edges[:-1]) - 1  # 0‑based

        # 2. Distribute target samples among bins (largest bins get extra)
        bin_counts = counts.copy()
        # Sort bin indices by count descending
        sorted_bins = np.argsort(bin_counts)[::-1]
        remaining = target_samples
        desired_counts = np.zeros(nbins, dtype=int)
        # each bin gets at most its count, distribute remaining as evenly as possible
        # Original logic: first round each bin gets floor(target/nbins)
        base = target_samples // nbins
        desired_counts[:] = base
        remaining = target_samples - base * nbins
        for b in sorted_bins[:remaining]:
            desired_counts[b] += 1
        # Cap at bin count
        desired_counts = np.minimum(desired_counts, bin_counts)

        # Avoid division by zero
        acceptance_prob = np.zeros(nbins)
        valid = bin_counts > 0
        acceptance_prob[valid] = desired_counts[valid] / bin_counts[valid]

        prob = acceptance_prob[bin_idx]
        # Random Bernoulli per point
        r = self.rng.random(n_points)
        selected = r < prob

        indices = np.where(selected)[0]
        # If we have too many or too few, adjust? The original script keeps all positive.
        # We'll just return exactly those.
        return coordinates[indices], values[indices]


class GradientHistogramSampling(BaseSamplingStrategy):
    """
    Gradient magnitude histogram‑based sampling.
    Uses same logic as HistogramSampling but on gradient magnitude instead of value.
    """
    def _compute_gradient_magnitude(self, coordinates: np.ndarray, values: np.ndarray) -> np.ndarray:
        # Reuse or copy from ImportanceSampling
        from scipy.spatial import cKDTree
        tree = cKDTree(coordinates)
        importance = np.zeros(len(values))
        for i in range(len(values)):
            distances, indices = tree.query(coordinates[i], k=min(6, len(values)))
            if len(indices) >= 3:
                local_coords = coordinates[indices]
                local_vals = values[indices]
                center = local_coords.mean(axis=0)
                local_coords_centered = local_coords - center
                try:
                    coeffs = np.linalg.lstsq(local_coords_centered, local_vals - local_vals.mean(), rcond=None)[0]
                    importance[i] = np.linalg.norm(coeffs)
                except:
                    importance[i] = 0
        return importance

    def sample(self, coordinates: np.ndarray, values: np.ndarray):
        grad_mag = self._compute_gradient_magnitude(coordinates, values)
        n_points = len(grad_mag)
        nbins = self.config.n_strata
        target_samples = self.config.n_samples

        counts, bin_edges = np.histogram(grad_mag, bins=nbins)
        bin_idx = np.digitize(grad_mag, bin_edges[:-1]) - 1

        base = target_samples // nbins
        desired_counts = np.full(nbins, base, dtype=int)
        remaining = target_samples - base * nbins
        sorted_bins = np.argsort(counts)[::-1]
        for b in sorted_bins[:remaining]:
            desired_counts[b] += 1
        desired_counts = np.minimum(desired_counts, counts)

        acceptance_prob = np.zeros(nbins)
        valid = counts > 0
        acceptance_prob[valid] = desired_counts[valid] / counts[valid]

        prob = acceptance_prob[bin_idx]
        r = self.rng.random(n_points)
        selected = r < prob
        indices = np.where(selected)[0]
        return coordinates[indices], values[indices]


class HistogramGradientSampling(BaseSamplingStrategy):
    """
    2D histogram of (value, gradient magnitude). Within each value bin,
    samples are distributed first to high‑gradient bins (descending order),
    then any remaining samples are spread evenly among all gradient bins.

    This matches the logic in `hist_grad_sampling_pymp` (without the final
    random‑gradient mixing).
    """
    def _compute_gradient_magnitude(self, coordinates: np.ndarray, values: np.ndarray) -> np.ndarray:
        # same as above
        from scipy.spatial import cKDTree
        tree = cKDTree(coordinates)
        importance = np.zeros(len(values))
        for i in range(len(values)):
            distances, indices = tree.query(coordinates[i], k=min(6, len(values)))
            if len(indices) >= 3:
                local_coords = coordinates[indices]
                local_vals = values[indices]
                center = local_coords.mean(axis=0)
                local_coords_centered = local_coords - center
                try:
                    coeffs = np.linalg.lstsq(local_coords_centered, local_vals - local_vals.mean(), rcond=None)[0]
                    importance[i] = np.linalg.norm(coeffs)
                except:
                    importance[i] = 0
        return importance

    def sample(self, coordinates: np.ndarray, values: np.ndarray):
        grad_mag = self._compute_gradient_magnitude(coordinates, values)
        n_points = len(values)
        nbins = self.config.n_strata
        target_samples = self.config.n_samples

        # 1D histogram of values to get desired counts per value bin
        val_counts, val_edges = np.histogram(values, bins=nbins)
        val_bin_idx = np.digitize(values, val_edges[:-1]) - 1

        # Distribute target samples among value bins (same as HistogramSampling)
        base = target_samples // nbins
        desired_val_counts = np.full(nbins, base, dtype=int)
        remaining = target_samples - base * nbins
        sorted_val_bins = np.argsort(val_counts)[::-1]
        for b in sorted_val_bins[:remaining]:
            desired_val_counts[b] += 1
        desired_val_counts = np.minimum(desired_val_counts, val_counts)

        # 2D histogram of (value_bin, grad_bin)
        grad_counts, grad_edges = np.histogram(grad_mag, bins=nbins)
        grad_bin_idx = np.digitize(grad_mag, grad_edges[:-1]) - 1

        hist_2d = np.histogram2d(values, grad_mag, bins=nbins)[0]  # (nbins, nbins)

        # For each value bin, distribute its desired samples among grad bins
        acceptance_prob_2d = np.zeros((nbins, nbins), dtype=np.float32)

        for v_bin in range(nbins):
            desired = desired_val_counts[v_bin]
            if desired == 0:
                continue
            grad_bin_counts = hist_2d[v_bin, :]   # counts per grad bin for this value bin
            total_in_vbin = grad_bin_counts.sum()
            if total_in_vbin == 0:
                continue

            # Stage 1: assign samples to grad bins in descending order of gradient magnitude
            # (but not by count – by grad bin index descending, i.e. high gradient)
            remaining_in_vbin = desired
            assigned = np.zeros(nbins, dtype=int)
            for g_bin in range(nbins-1, -1, -1):
                if remaining_in_vbin <= 0:
                    break
                take = min(remaining_in_vbin, grad_bin_counts[g_bin])
                assigned[g_bin] = take
                remaining_in_vbin -= take

            # Stage 2: if still remaining, spread them evenly across all grad bins
            if remaining_in_vbin > 0:
                # add one by one to bins that have capacity left
                order = np.arange(nbins)   # any order
                for g_bin in order:
                    if remaining_in_vbin <= 0:
                        break
                    if assigned[g_bin] < grad_bin_counts[g_bin]:
                        assigned[g_bin] += 1
                        remaining_in_vbin -= 1

            # Compute acceptance probability per 2D bin
            for g_bin in range(nbins):
                if grad_bin_counts[g_bin] > 0:
                    acceptance_prob_2d[v_bin, g_bin] = assigned[g_bin] / grad_bin_counts[g_bin]

        # For each point, get its acceptance probability
        prob = acceptance_prob_2d[val_bin_idx, grad_bin_idx]
        r = self.rng.random(n_points)
        selected = r < prob
        indices = np.where(selected)[0]
        return coordinates[indices], values[indices]




class SLICSampling(BaseSamplingStrategy):
    """
    SLIC (Simple Linear Iterative Clustering) superpixel sampling for 2D grids.

    Steps:
      1. Infer grid dimensions from unique coordinates.
      2. Reshape scalar values into a 2D image.
      3. Apply SLIC to obtain superpixel labels.
      4. For each superpixel, compute centroid (mean of original coordinates)
         and mean value.
      5. If number of superpixels != n_samples, adjust:
         - If more superpixels: merge the smallest ones (by area) until target reached.
         - If fewer superpixels: add random samples from the original dataset.
      6. Return sampled points and values.
    """
    def sample(self, coordinates: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if coordinates.shape[1] != 2:
            raise ValueError("SLIC sampling is only implemented for 2D coordinates.")

        # 1. Infer grid dimensions
        x_vals = coordinates[:, 0]
        y_vals = coordinates[:, 1]
        # Assuming regular grid: use unique sorted coordinates
        x_unique = np.sort(np.unique(x_vals))
        y_unique = np.sort(np.unique(y_vals))
        nx, ny = len(x_unique), len(y_unique)
        expected_pts = nx * ny
        if expected_pts != coordinates.shape[0]:
            raise ValueError("Coordinates do not form a regular grid (unique counts mismatch).")

        # Reshape values into 2D image
        img = values.reshape(ny, nx).T   # shape (nx, ny) or (ny, nx)? Let's make it (ny, nx) for image
        # SLIC expects (height, width) array. We'll reshape to (ny, nx)
        img_2d = values.reshape(ny, nx)  # now rows = y, cols = x

        # 2. Run SLIC
        # Determine number of superpixels: heuristic ~ target samples / 2 (to allow merging)
        n_segments = max(2, int(self.config.n_samples * 1.5))
        compactness = 10.0  # default, can be made configurable
        segments = slic(img_2d, n_segments=n_segments, compactness=compactness, start_label=0)

        # 3. Compute per‑segment representatives
        unique_segments = np.unique(segments)
        seg_centroids = []
        seg_means = []
        for seg_id in unique_segments:
            mask = (segments == seg_id)
            # Get indices of points belonging to this superpixel in the original dataset
            # We need to map from 2D indices to linear index
            rows, cols = np.where(mask)
            # Convert to original coordinate values
            xs = x_unique[cols]
            ys = y_unique[rows]
            centroid_x = np.mean(xs)
            centroid_y = np.mean(ys)
            mean_val = np.mean(img_2d[mask])
            seg_centroids.append([centroid_x, centroid_y])
            seg_means.append(mean_val)

        seg_centroids = np.array(seg_centroids)
        seg_means = np.array(seg_means)
        current_n = len(seg_centroids)

        # 4. Adjust to exact target
        target = self.config.n_samples
        if current_n > target:
            # Need to merge superpixels: repeatedly merge the smallest (by area) adjacent?
            # Simplified: randomly sample `target` centroids
            chosen = self.rng.choice(current_n, size=target, replace=False)
            sampled_coords = seg_centroids[chosen]
            sampled_vals = seg_means[chosen]
        elif current_n < target:
            # Need more points: add random samples from the original dataset
            extra = target - current_n
            all_indices = np.arange(coordinates.shape[0])
            extra_indices = self.rng.choice(all_indices, size=extra, replace=False)
            extra_coords = coordinates[extra_indices]
            extra_vals = values[extra_indices]
            sampled_coords = np.vstack([seg_centroids, extra_coords])
            sampled_vals = np.concatenate([seg_means, extra_vals])
            # Shuffle to mix
            perm = self.rng.permutation(len(sampled_coords))
            sampled_coords = sampled_coords[perm]
            sampled_vals = sampled_vals[perm]
        else:
            sampled_coords = seg_centroids
            sampled_vals = seg_means

        # Indices: find nearest neighbor in original dataset
        tree = cKDTree(coordinates)
        _, indices = tree.query(sampled_coords)
        return sampled_coords, sampled_vals





def get_sampling_strategy(config: SamplingConfig) -> BaseSamplingStrategy:
    """Factory function to get the appropriate sampling strategy."""
    strategies = {
        SamplingType.RANDOM:            RandomSampling,
        SamplingType.UNIFORM_GRID:      UniformGridSampling,
        SamplingType.STRATIFIED:        StratifiedSampling,
        SamplingType.IMPORTANCE:        ImportanceSampling,
        SamplingType.EDGE_BIASED:       EdgeBiasedSampling,
        SamplingType.RANDOM_STRATIFIED: RandomStratifiedSampling,
        SamplingType.LATIN_HYPERCUBE:   LatinHypercubeSampling,
        SamplingType.GRID_AGGREGATE:    GridAggregateSampling,   
        SamplingType.HISTOGRAM: HistogramSampling,
        SamplingType.GRADIENT_HISTOGRAM: GradientHistogramSampling,
        SamplingType.HISTOGRAM_GRADIENT: HistogramGradientSampling,
        SamplingType.SLIC: SLICSampling,
    }

    strategy_class = strategies.get(config.sampling_type)
    if strategy_class is None:
        raise ValueError(f"Unknown sampling type: {config.sampling_type}")

    return strategy_class(config)