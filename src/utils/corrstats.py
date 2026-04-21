# Modules -----------------------------------------------------------------------------------------------------------------#
import numpy  as np

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing               import Optional, Union, Tuple, Literal
from sklearn.neighbors    import BallTree, KDTree
from sklearn.linear_model import LinearRegression
from scipy.stats          import pearsonr, spearmanr
from scipy.special        import digamma
from joblib               import Parallel, delayed

# Partial Correlation Coefficient (PCC) Estimator with Bootstrap Uncertainty ----------------------------------------------#
def pcc_estimator(X : np.ndarray, y : np.ndarray, corr_metric : Literal['pearson', 'spearman'] = 'pearson',
                  n_bootstrap : int           = 1000,
                  n_jobs      : int           = 8,
                  random_seed : Optional[int] = 42
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    ________________________________________________________________________________________________________________________
    Partial Correlation Coefficient (PCC) estimator with bootstrap uncertainty for a feature matrix X against a target y.
    Each feature's partial correlation controls for all remaining features through linear regression residuals.
    Computation across features is parallelized with joblib.
    ________________________________________________________________________________________________________________________
    Parameters:
    -> X           (np.ndarray)    : Feature matrix. Shape (n_samples, n_features). Mandatory.
    -> y           (np.ndarray)    : Target vector. Shape (n_samples,). Mandatory.
    -> corr_metric (str)           : Correlation metric ('pearson' or 'spearman'). Default 'pearson'.
    -> n_bootstrap (int)           : Number of bootstrap resamples for uncertainty estimation. Default 1000.
    -> n_jobs      (int)           : Number of parallel workers. -1 uses all available CPUs. Default 8.
    -> random_seed (int, optional) : Base seed for reproducibility. Each worker receives (random_seed + i).
                                     Set None to disable seeding. Default 42.
    ________________________________________________________________________________________________________________________
    Returns:
    -> pcc_values (np.ndarray) : Partial correlation coefficient per feature, shape (n_features,).
    -> pcc_errors (np.ndarray) : Bootstrap standard deviation (uncertainty) per feature, shape (n_features,).
    ________________________________________________________________________________________________________________________
    Notes:
        For each feature Xi the partial correlation is computed as:

                1. Regress X_{-i} -> Xi  to obtain residuals  res_x = Xi  - X_{-i} * beta_x
                2. Regress X_{-i} -> y   to obtain residuals  res_y = y   - X_{-i} * beta_y
                3. PCC(Xi, y | X_{-i}) = corr(res_x, res_y)

        When n_features == 1, no regression is applied and the regular correlation is returned directly.
        Bootstrap resampling is performed on the residuals, which is valid since the regression step is
        independent of the bootstrap draw.
    ________________________________________________________________________________________________________________________
    """
    # [Helper] Pearson or Spearman correlation between two 1-D arrays ----------------------------------------------------#
    def _compute_correlation(a: np.ndarray, b: np.ndarray) -> float:
        if corr_metric == 'pearson':
            return pearsonr(a, b)[0]
        return spearmanr(a, b)[0]

    # [Helper] PCC and bootstrap uncertainty for a single feature index --------------------------------------------------#
    def _compute_feature(i: int, seed: Optional[int]) -> Tuple[float, float]:
        if seed is not None:
            np.random.seed(seed)

        idx_other = [j for j in range(n_features) if j != i]

        # With a single feature, partial correlation degenerates to regular correlation
        if len(idx_other) == 0:
            return _compute_correlation(X[:, i], y), 0.0

        # Regress out all other features from Xi and y to obtain marginal residuals
        reg_x = LinearRegression(fit_intercept=True).fit(X[:, idx_other], X[:, i])
        res_x = X[:, i] - reg_x.predict(X[:, idx_other])

        reg_y = LinearRegression(fit_intercept=True).fit(X[:, idx_other], y)
        res_y = y - reg_y.predict(X[:, idx_other])

        # PCC is the correlation between the two sets of residuals
        pcc_value = _compute_correlation(res_x, res_y)

        # Bootstrap uncertainty on residuals -----------------------------------------------------------------------------#
        bootstrap_pccs = np.full(n_bootstrap, np.nan)
        for b in range(n_bootstrap):
            boot_idx = np.random.randint(0, n_samples, size=n_samples)
            try:
                bootstrap_pccs[b] = _compute_correlation(res_x[boot_idx], res_y[boot_idx])
            except Exception:
                pass  # Leave as nan; filtered out below

        # Standard deviation over valid bootstrap draws; fall back to 0.0 if all failed
        valid     = bootstrap_pccs[~np.isnan(bootstrap_pccs)]
        pcc_error = np.std(valid) if len(valid) > 0 else 0.0

        return pcc_value, pcc_error

    # Coerce inputs to plain 2-D float arrays
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    
    n_samples, n_features = X.shape

    # Assign a unique reproducible seed per worker
    seeds = [random_seed + i if random_seed is not None else None for i in range(n_features)]

    # Parallelized computation across features
    outputs = Parallel(n_jobs=n_jobs)(delayed(_compute_feature)(i, seeds[i]) for i in range(n_features))

    pcc_values = np.array([v for v, _ in outputs])
    pcc_errors = np.array([e for _, e in outputs])

    return pcc_values, pcc_errors

# Kraskov, Stögbauer and Grassberger (KSG) Mutual Information Estimator ---------------------------------------------------#
def ksg_mi_estimator(x : Union[list, np.ndarray], y : Union[list, np.ndarray], z : Optional[Union[list, np.ndarray]] = None,
                     k      : int               = 3,
                     base   : Union[int, float] = 2,
                     alpha  : float             = 0,
                     intest : float             = 1e-10
                     ) -> float:
    """
    ________________________________________________________________________________________________________________________
    Kraskov, Stögbauer and Grassberger (KSG) estimator of Mutual Information I(X;Y) between two variables
    ________________________________________________________________________________________________________________________
    Parameters:
    -> x      (array-like)           : First variable. Shape (n_samples,) or (n_samples, d_x). Mandatory.
    -> y      (array-like)           : Second variable. Shape (n_samples,) or (n_samples, d_y). Mandatory.
    -> z      (array-like, optional) : Conditioning variable. If provided, computes CMI I(X;Y|Z) instead of MI
                                       I(X;Y). Shape (n_samples,) or (n_samples, d_z). Default None.
    -> k      (int)                  : Number of nearest neighbors used for the density estimation. Larger values
                                       reduce variance but increase bias. Default 3.
    -> base   (int/float)            : Logarithm base controlling the unit of the output. Default 2 (bits).
                                       Use np.e for nats or 10 for hartleys.
    -> alpha  (float)                : Local Non-Uniformity Correction (LNC) strength parameter. Mitigates
                                       overestimation in regions of high local density variation. Set > 0 to enable.
                                       Default 0 (disabled). Only applied when z is None.
    -> intest (float)                : Intensity of the uniform noise added to all inputs before tree construction,
                                       used to break degeneracy from duplicate values (ties). Default 1e-10.
    ________________________________________________________________________________________________________________________
    Returns:
    -> mi (float) [bits/nats] : Estimated mutual information I(X;Y) if z is None, or conditional mutual information
                                I(X;Y|Z) otherwise. Units are determined by the chosen logarithm base.
    ________________________________________________________________________________________________________________________
    Notes:
        KSG estimator from: Kraskov, Stögbauer & Grassberger (2004), Phys. Rev. E 69, 066138.
        The unconditional case follows Algorithm 1 (Eq. 8) from the paper. The conditional case extends the
        same k-nearest neighbor framework to the joint space (X, Y, Z).

        The Local Non-Uniformity Correction (LNC) follows: Gao et al. (2015), IEEE Trans. Inf. Theory,
        and is applied only in the unconditional case when alpha > 0. The correction penalizes neighborhoods
        where the data distribution is highly non-uniform in the PCA-aligned bounding box.

        Small uniform noise (intest) is added to all variables before tree construction to ensure that no
        two points are at zero distance in the marginal spaces, which would produce undefined digamma values.
    ________________________________________________________________________________________________________________________
    References:
        - Implementation from: https://github.com/gregversteeg/NPEET/tree/master
    ________________________________________________________________________________________________________________________
    """
    # [Helper] Build the tree estimator based on actual number of dimensions ------------------------------------------------#
    def _build_tree(points: np.ndarray) -> Union[BallTree, KDTree]:
        # BallTree scales better in high dimensions; KDTree is faster below ~20 dims
        if points.shape[1] >= 20:
            return BallTree(points, metric="chebyshev")
        
        # Else, implement the KDTree
        return KDTree(points, metric="chebyshev")

    # [Helper] Average digamma of neighbor counts in a marginal space within epsilon-ball radii ----------------------#
    def _avgdigamma(points: np.ndarray, dvec: np.ndarray) -> float:
        # Build a marginal tree and shrink the radii slightly to enforce strict inequality (open ball)
        tree       = _build_tree(points)
        dvec       = dvec - 1e-15
        num_points = tree.query_radius(points, dvec, count_only=True)
        
        # Return the mean of psi(n_x) as required by the KSG estimator formula
        return np.mean(digamma(num_points))
    
    # [Helper] Local Non-Uniformity Correction (LNC) — reduces overestimation in non-uniform density regions ----------#
    def _lnc_correction(tree: Union[BallTree, KDTree], points: np.ndarray, k: int, alpha: float) -> float:
        e        = 0
        n_sample = points.shape[0]
        for point in points:
            # Retrieve k-nearest neighbor indices in the joint space (Chebyshev / max-norm)
            knn        = tree.query(point[None, :], k=k + 1, return_distance=False)[0]
            knn_points = points[knn]
            
            # Center the neighborhood at the query point for PCA alignment
            knn_points = knn_points - knn_points[0]
            
            # Build the scatter matrix and extract eigen vectors via eigh (symmetric, guarantees real output)
            covr = knn_points.T @ knn_points / k
            _, v = np.linalg.eigh(covr)
            
            # Volume of the PCA-aligned bounding box (log-sum over projected extents)
            V_rect = np.log(np.abs(knn_points @ v).max(axis=0)).sum()
            
            # Volume of the original Chebyshev ball (log-sum over coordinate extents)
            log_knn_dist = np.log(np.abs(knn_points).max(axis=0)).sum()

            # Apply correction only when the neighborhood is locally non-uniform (PCA box is significantly smaller)
            # Guard (log_knn_dist - V_rect) > 0 to protect against rare numerical precision issues
            if V_rect < log_knn_dist + np.log(alpha) and log_knn_dist > V_rect:
                e += (log_knn_dist - V_rect) / n_sample
        return e

    # Coerce inputs to 2-D numpy arrays (n_samples x d)
    x, y = np.asarray(x), np.asarray(y)
    x, y = x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1)
    
    # Add small uniform noise to all inputs to break ties and avoid ill-defined digamma(0) evaluations
    rng     = np.random.default_rng()
    x_noise = x + intest * rng.random(x.shape)
    y_noise = y + intest * rng.random(y.shape)
    
    # Collect perturbed marginals for joint stacking; append z if conditioning is requested
    points  = [x_noise, y_noise]
    z_noise = None
    if z is not None:
        z       = np.asarray(z)
        z       = z.reshape(z.shape[0], -1)
        z_noise = z + intest * rng.random(z.shape)
        points.append(z_noise)
    
    # Build the joint space and find the k-th neighbor distance per point (epsilon radii)
    points = np.hstack(points)
    tree   = _build_tree(points)
    dvec   = tree.query(points, k=k + 1)[0][:, k]
    
    # Unconditional MI: I(X;Y) = psi(k) + psi(N) - <psi(n_x)> - <psi(n_y)>
    if z is None:
        a, b, c, d = (_avgdigamma(x_noise, dvec), _avgdigamma(y_noise, dvec), digamma(k), digamma(len(x)))
        
        # Optional LNC correction additive term on psi(N)
        if alpha > 0:
            d += _lnc_correction(tree, points, k, alpha)
    
    # Conditional MI: I(X;Y|Z) = psi(k) + <psi(n_z)> - <psi(n_xz)> - <psi(n_yz)>
    else:
        xz = np.c_[x_noise, z_noise]
        yz = np.c_[y_noise, z_noise]
        a, b, c, d = (_avgdigamma(xz, dvec), _avgdigamma(yz, dvec), _avgdigamma(z_noise, dvec), digamma(k))
        
    return (-a - b + c + d) / np.log(base)

#--------------------------------------------------------------------------------------------------------------------------#