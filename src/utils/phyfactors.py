# Modules -----------------------------------------------------------------------------------------------------------------#
import numpy         as np
import astropy.units as u

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing  import Optional, Union, Tuple
from astropy import constants as c

# Computation of crossing times -------------------------------------------------------------------------------------------#
def crossing_time(hm_radius: Union[float, np.ndarray], v_disp: Optional[Union[float, np.ndarray]] = None, 
                  mtot: Optional[Union[float, np.ndarray]] = None) -> Tuple[np.ndarray, u.Unit]:
    """
    ________________________________________________________________________________________________________________________
    Calculate the crossing time of a star cluster. (Unit conversion is handled by Astropy)
    ________________________________________________________________________________________________________________________
    Parameters:
        - hm_radius (Union[float, np.ndarray])           [pc]   : Half-mass radius of the star cluster. Mandatory.
        - v_disp    (Optional[Union[float, np.ndarray]]) [km/s] : Velocity dispersion of the star cluster. Optional.
        - mtot      (Optional[Union[float, np.ndarray]]) [Msun] : Total mass of the star cluster. 
                                                                  Mandatory if v_disp is None.
    ________________________________________________________________________________________________________________________
    Returns:
        - t_cross, t_unit (Tuple[np.ndarray, astropy.unit]) [Myr] : Crossing time of the star cluster (time that takes a 
                                                                    star to travel across the whole system under the 
                                                                    influence of the gravitational field of the cluster).
                                                                    Returns array with values and astropy temporal unit.
    ________________________________________________________________________________________________________________________
    Notes:
        Formula from Spitzer (1987). Standard approximation:  

                t_cross = (Rh / v_disp)
        
        If v_disp is None we approximate the velocity dispersion through virial theorem as: 
        
                v_disp = sqrt((3*G*Mtot)/(5*Rh)) 
        
        With G the gravitational constant, Mtot the total mass of the cluster and Rh the half-mass radius which is valid 
        for a homogeneous sphere.
    ________________________________________________________________________________________________________________________
    """   
    # Convert all inputs to arrays for unified processing
    hm_radius = np.atleast_1d(hm_radius)
    
    # Define working units
    hm_unit     = u.pc
    time_unit   = u.Myr
    v_disp_unit = u.km / u.s
    mtot_unit   = u.solMass

    # Input validation
    if np.any(hm_radius <= 0):
        raise ValueError("Half-mass radius must be positive")
    
    # Handle velocity dispersion
    if v_disp is None:
        if mtot is None:
            raise ValueError("Total mass must be provided when velocity dispersion is not given")

        # Convert mtot to array if applicable
        mtot      = np.atleast_1d(mtot)
        
        if np.any(mtot <= 0):
            raise ValueError("Total mass must be positive")
        
        # Calculate velocity dispersion using virial theorem
        G_val = c.G.to(hm_unit**3 / (mtot_unit * time_unit**2)).value
        v_val = np.sqrt((3 * G_val * mtot) / (5 * hm_radius))
        
        # Retrieve in proper units
        v_disp_myrs_pc = v_val * (hm_unit / time_unit)
        v_disp         = v_disp_myrs_pc.to(v_disp_unit).value

    else:
        v_disp = np.atleast_1d(v_disp)
        if np.any(v_disp <= 0):
            raise ValueError("Velocity dispersion must be positive")
    
    # Calculate crossing time
    t_cross = (hm_radius / v_disp)

    return t_cross, time_unit

# Computation of number density within half-mass radius ------------------------------------------------------------------#
def rho_at_rh(n_stars: Union[int, np.ndarray], hm_radius: Union[float, np.ndarray]) -> Tuple[np.ndarray, u.Unit]:
    """
    ________________________________________________________________________________________________________________________
    Calculate the number density within the half-mass radius of a star cluster. (Unit conversion is handled by Astropy)
    ________________________________________________________________________________________________________________________
    Parameters:
        n_stars   (Union[int, np.ndarray])          [#]    : Number of stars in the cluster. Mandatory.
        hm_radius (Union[float, np.ndarray])        [pc]   : Half-mass radius of the star cluster. Mandatory.
    ________________________________________________________________________________________________________________________
    Returns:
        n_density, density_unit (Tuple[np.ndarray, u.Unit]) [pc^-3]: Number density within the half-mass radius of the star
                                                                     cluster, assuming a uniform distribution of stars 
                                                                     across the cluster volume.
                                                                     Returns array with values and astropy temporal unit.
    ________________________________________________________________________________________________________________________
    Notes:
        Formula assumes a uniform distribution of stars across the cluster volume. The number density is approximated as:
        
                n_density = (3 * n_stars) / (8 * pi * hm_radius^3)
        
        This approximation is valid for a homogeneous sphere and provides a reasonable estimate of the stellar number 
        density within the half-mass radius of the cluster.
    ________________________________________________________________________________________________________________________
    """
    # Convert to arrays for validation
    n_stars   = np.atleast_1d(n_stars)
    hm_radius = np.atleast_1d(hm_radius)

    # Define units
    radius_unit  = u.pc
    density_unit = radius_unit**(-3)

    # Input validation
    if np.any(n_stars <= 0):
        raise ValueError("Number of stars in the cluster must be greater than zero")
    if np.any(hm_radius <= 0):
        raise ValueError("Half-mass radius must be positive")
    
    # Vectorized calculation
    n_density = (3 * n_stars) / (8 * np.pi * hm_radius**3)
        
    # Validate that result is reasonable (should be positive)
    if np.any(n_density <= 0):
        raise ValueError("Number density calculation resulted in non-positive value")
        
    return n_density, density_unit

# Computation of relaxation time ------------------------------------------------------------------------------------------#
def relaxation_time(n_stars: Union[int, np.ndarray], hm_radius: Union[float, np.ndarray], 
                    v_disp: Optional[Union[float, np.ndarray]] = None, 
                    mtot  : Optional[Union[float, np.ndarray]] = None) -> Tuple[np.ndarray, u.Unit]:
    """
    ________________________________________________________________________________________________________________________
    Calculate the relaxation time of a star cluster. (Unit conversion is handled by Astropy)
    ________________________________________________________________________________________________________________________
    Parameters:
        n_stars   (Union[int, np.ndarray])             [#]    : Number of stars in the cluster. Mandatory
        hm_radius (Union[float, np.ndarray])           [pc]   : Half-mass radius of the star cluster. Mandatory.
        v_disp    (Optional[Union[float, np.ndarray]]) [km/s] : Velocity dispersion of the star cluster. Optional.
        mtot      (Optional[Union[float, np.ndarray]]) [Msun] : Total mass of the star cluster. Mandatory if v_disp is None.
    ________________________________________________________________________________________________________________________
    Returns:
        t_relax, t_unit (Tuple[np.ndarray, u.Unit]) [Myr]  : Relaxation time of the star cluster (time required for stellar 
                                                             encounters to significantly alter the velocities of stars and 
                                                             drive the system toward energy equipartition).
                                                             Returns array with values and astropy temporal unit.
    ________________________________________________________________________________________________________________________
    Notes:
        Formula from Binney & Tremaine, Galactic Dynamics, Second Edition (Eq 1.38). The relaxation time is given by:
        
                t_relax = 0.1 * (N * t_cross) / ln(N)
        
        Where N is the number of stars, t_cross is the crossing time. Crossing time is computed internally using 
        crossing_time() function.
    ________________________________________________________________________________________________________________________
    """
    # Convert to arrays for validation
    n_stars = np.atleast_1d(n_stars)
    
    # Input validation
    if np.any(n_stars <= 0):
        raise ValueError("Number of stars in the cluster must be greater than zero")
    if np.any(n_stars < 10):
        raise ValueError("Relaxation time calculation requires at least 10 stars for meaningful results")
    
    # Compute crossing time
    t_cross, time_unit = crossing_time(hm_radius, v_disp, mtot)
    
    
    # Vectorized calculation
    t_relax = 0.1 * (n_stars * t_cross) / np.log(n_stars)
    
    return t_relax, time_unit

# Computation of core collapse time ---------------------------------------------------------------------------------------#
def core_collapse_time(m_mean: Union[float, np.ndarray], m_max: Union[float, np.ndarray], n_stars: Union[int, np.ndarray], 
                       hm_radius: Union[float, np.ndarray], 
                       v_disp   : Optional[Union[float, np.ndarray]] = None, 
                       mtot     : Optional[Union[float, np.ndarray]] = None) -> Tuple[np.ndarray, u.Unit]:
    """
    ________________________________________________________________________________________________________________________
    Calculate the core collapse time of a star cluster. (Unit conversion is handled by Astropy)
    ________________________________________________________________________________________________________________________
    Parameters:
        m_mean    (Union[float, np.ndarray])           [Msun] : Mean mass per star. Mandatory.
        m_max     (Union[float, np.ndarray])           [Msun] : Maximum stellar mass (considering the initial range of 
                                                                stellar masses). Mandatory.
        n_stars   (Union[int, np.ndarray])             [#]    : Number of stars in the cluster. Mandatory
        hm_radius (Union[float, np.ndarray])           [pc]   : Half-mass radius of the star cluster. Mandatory.
        v_disp    (Optional[Union[float, np.ndarray]]) [km/s] : Velocity dispersion of the star cluster. Optional.
        mtot      (Optional[Union[float, np.ndarray]]) [Msun] : Total mass of the star cluster. Mandatory if v_disp is None.
    ________________________________________________________________________________________________________________________
    Returns:
        t_cc, t_unit (Tuple[np.ndarray, u.Unit]) [Myr]: Core collapse time of the star cluster, which accounts for the 
                                                        presence of massive stars. The time is scaled by the ratio of the 
                                                        mean stellar mass to the maximum stellar mass, reflecting the 
                                                        accelerated collapse driven by mass segregation.
                                                        Returns array with values and astropy temporal unit.
    ________________________________________________________________________________________________________________________
    Notes:
        Formula from Portegies & McMillan (2002) accounting for mass segregation: 
        
                t_cc = 3.3 * (m_mean / m_max) * t_relax
        
        Where t_relax is the relaxation time. Relaxation time is computed internally using relaxation_time() function.
    ________________________________________________________________________________________________________________________
    """
    # Convert to arrays for validation
    m_mean = np.atleast_1d(m_mean)
    m_max  = np.atleast_1d(m_max)
    
    # Input validation
    if np.any(m_mean <= 0):
        raise ValueError("mean mass per star must be greater than zero")
    if np.any(m_max <= 0):
        raise ValueError("max mass must be greater than zero")
    
    # Vectorized calculation
    t_relax, t_unit = relaxation_time(n_stars, hm_radius, v_disp, mtot)
    t_cc = 3.3 * (m_mean / m_max) * t_relax
    
    return t_cc, t_unit

# Computation of safronov number ------------------------------------------------------------------------------------------#
def safronov_num(mass_per_star: Union[float, np.ndarray], v_disp: Union[float, np.ndarray], 
                 stellar_radius: Union[float, np.ndarray] = 1.0) -> Tuple[np.ndarray, u.Unit]:
    """
    ________________________________________________________________________________________________________________________
    Calculate the Safronov number of a star cluster. (Unit conversion is handled by Astropy)
    ________________________________________________________________________________________________________________________
    Parameters:
        mass_per_star   (Union[float, np.ndarray])  [Msun] : Typical stellar mass per star in the cluster. Mandatory.
        v_disp          (Union[float, np.ndarray])  [km/s] : Velocity dispersion of the star cluster. Mandatory.
        stellar_radius  (Union[float, np.ndarray])  [Rsun] : Stellar radius. Default is 1.0 Rsun.
    ________________________________________________________________________________________________________________________
    Returns:
        theta (Union[float, np.ndarray]) [#]: Safronov number, a dimensionless parameter that measures the importance of 
                                              gravitational focusing in stellar encounters. Values >> 1 indicate strong 
                                              gravitational focusing, while values << 1 indicate weak focusing.
                                              Returns array with values and astropy temporal unit.
    ________________________________________________________________________________________________________________________
    Notes:
        Formula adapted from Binney & Tremaine, Galactic Dynamics, Second Edition (Eq. 7.195b). The Safronov number is 
        given by:
        
                theta = 9.54 * (M/Msun) * (Rsun/R) * (100 km/s / v_disp)^2
        
        Where M is the stellar mass, R is the stellar radius, and v_disp is the velocity dispersion. This parameter 
        determines whether encounters are dominated by gravitational focusing (theta >> 1) or geometric 
        cross-section (theta << 1).
    ________________________________________________________________________________________________________________________
    """
    # Convert to arrays for calculation
    mass_per_star  = np.asarray(mass_per_star)
    v_disp         = np.asarray(v_disp)
    stellar_radius = np.asarray(stellar_radius)
    
    # Input validation
    if np.any(mass_per_star <= 0):
        raise ValueError("Stellar mass must be positive")
    if np.any(v_disp <= 0):
        raise ValueError("Velocity dispersion must be positive")
    if np.any(stellar_radius <= 0):
        raise ValueError("Stellar radius must be positive")
    
    # Define units 
    mass_per_star_unit =  u.solMass
    v_disp_unit        =  u.km / u.s
    stellar_radius_u   =  u.solRad
        
    # Calculate Safronov number using astropy units
    solar_mass         = 1.0 * u.solMass
    solar_radius       = 1.0 * u.solRad
    reference_velocity = 100.0 * u.km / u.s
    
    theta      = 9.54 * (mass_per_star / solar_mass.value) * (solar_radius.value / stellar_radius) * (reference_velocity.value / v_disp)**2
    theta_unit = (mass_per_star_unit / solar_mass.unit) * (solar_radius.unit/stellar_radius_u) * (reference_velocity.unit / v_disp_unit)**2
    
    # Ensure the result is dimensionless
    theta_unit = theta_unit.decompose()
    
    # Assert that the result is dimensionless
    assert theta_unit.is_unity(), f"Safronov number must be dimensionless, got {theta_unit}"
    
    # Validate that result is reasonable (should be positive)
    if np.any(theta <= 0):
        raise ValueError("Safronov number calculation resulted in non-positive value")
    
    return theta, theta_unit

# Computation of collision time -------------------------------------------------------------------------------------------#
def collision_time(hm_radius: Union[float, np.ndarray], n_stars: Union[int, np.ndarray], 
                   mass_per_star  : Union[float, np.ndarray], 
                   v_disp         : Union[float, np.ndarray], 
                   stellar_radius : Union[float, np.ndarray] = 1.0) -> Tuple[np.ndarray, u.Unit]:
    """
    ________________________________________________________________________________________________________________________
    Calculate the collision time of a star cluster. (Unit conversion is handled by Astropy)
    ________________________________________________________________________________________________________________________
    Parameters:
        hm_radius      (Union[float, np.ndarray]) [pc]   : Half-mass radius of the star cluster. Mandatory.
        n_stars        (Union[int, np.ndarray])   [#]    : Number of stars in the cluster. Mandatory.
        mass_per_star  (Union[float, np.ndarray]) [Msun] : Typical stellar mass per star in the cluster. Mandatory.
        v_disp         (Union[float, np.ndarray]) [km/s] : Velocity dispersion of the star cluster. Mandatory.
        stellar_radius (Union[float, np.ndarray]) [Rsun] : Stellar radius. Default is 1.0 Rsun.
    ________________________________________________________________________________________________________________________
    Returns:
        t_coll, t_unit (np.ndarray, u.Unit]) [Myr]: Collision time of the star cluster (average time 
                                                    between stellar collisions, accounting for 
                                                    gravitational focusing effects).
                                                    Returns array with values and astropy temporal unit.
    ________________________________________________________________________________________________________________________
    Notes:
        Formula from Binney & Tremaine, Galactic Dynamics, Second Edition (Eq. 7.195). The expression includes 
        gravitational focusing. The collision time is given by:

                t_coll = (16 * sqrt(pi) * n * v_disp * stellar_radius**2 * (1+ theta))**(-1)
        
        With n the number density within the half mass radius, v_disp the velocity dispersion, stellar_radius the stellar 
        radius, and theta the Safronov number. The number density is being approximated as:

                n = (3 * n_stars ) / (8 * pi * hm_radius**3),

        Assuming a uniform distribution across the cluster. Safronov number is computed internally using 
        safronov_num() function.
    ________________________________________________________________________________________________________________________
    """
    # Convert to arrays for validation
    hm_radius      = np.atleast_1d(hm_radius)
    n_stars        = np.atleast_1d(n_stars)
    mass_per_star  = np.atleast_1d(mass_per_star)
    v_disp         = np.atleast_1d(v_disp)
    stellar_radius = np.atleast_1d(stellar_radius)
    
    # Define units
    time_unit = u.Myr
    
    # Input validation
    if np.any(hm_radius <= 0):
        raise ValueError("Half-mass radius must be positive")
    if np.any(n_stars <= 0):
        raise ValueError("Number of stars must be positive")
    if np.any(mass_per_star <= 0):
        raise ValueError("Stellar mass must be positive")
    if np.any(v_disp <= 0):
        raise ValueError("Velocity dispersion must be positive")
    if np.any(stellar_radius <= 0):
        raise ValueError("Stellar radius must be positive")
    
    # Calculate number density within half-mass radius
    n_density, density_unit = rho_at_rh(n_stars, hm_radius)
    
    # Calculate Safronov number
    theta, theta_unit = safronov_num(mass_per_star, v_disp, stellar_radius)
    
    # Convert stellar radius from solar radii to pc
    solar_radius_pc   = c.R_sun.to(u.pc).value
    stellar_radius_pc = stellar_radius * solar_radius_pc
    
    # Convert v_disp from km/s to pc/Myr
    v_disp_pc_myr = (v_disp * u.km / u.s).to(u.pc / time_unit).value
    
    # Calculate collision time: t_coll = 1 / (16 * sqrt(pi) * n * v_disp * R_star^2 * (1 + theta))
    t_coll = 1.0 / (16 * np.sqrt(np.pi) * n_density * v_disp_pc_myr * stellar_radius_pc**2 * (1 + theta))
    
    # Validate that result is reasonable (should be positive)
    if np.any(t_coll <= 0):
        raise ValueError("Collision time calculation resulted in non-positive value")
    
    return t_coll, time_unit

# Critical Mass -----------------------------------------------------------------------------------------------------------#
def critical_mass(hm_radius: Union[float, np.ndarray], mass_per_star: Union[float, np.ndarray], 
                  cluster_age    : Union[float, np.ndarray], 
                  v_disp         : Union[float, np.ndarray], 
                  stellar_radius : Union[float, np.ndarray] = 1.0) -> Tuple[np.ndarray, u.Unit]:
    """
    ________________________________________________________________________________________________________________________
    Calculate the critical mass of a star cluster. (Unit conversion is handled by Astropy)
    ________________________________________________________________________________________________________________________
    Parameters:
        hm_radius       (Union[float, np.ndarray]) [pc]    : Half-mass radius of the star cluster. Mandatory.
        mass_per_star   (Union[float, np.ndarray]) [Msun]  : Typical stellar mass per star in the cluster. Mandatory.
        cluster_age     (Union[float, np.ndarray]) [Myr]   : Age of the cluster (or time since beginning of the simulation).
                                                             Mandatory.
        v_disp          (Union[float, np.ndarray]) [km/s]  : Velocity dispersion of the cluster. Mandatory.
        stellar_radius  (Union[float, np.ndarray]) [Rsun]  : Typical stellar radius. Default is 1.0 Rsun.
    ________________________________________________________________________________________________________________________
    Returns:
        m_crit, m_unit (Tuple[Union[float, np.ndarray], u.Unit]) [Msun] : Critical mass of the star cluster, beyond which 
                                                                          the system may undergo runaway collisions or core 
                                                                          collapse. Returns array with values and astropy 
                                                                          mass unit.
    ________________________________________________________________________________________________________________________
    Notes:
        Formula from Vergara et al. (2023), also present in Vergara et al (2024) However we do not take into account the
        term related to an external potential due to the presence of gas in the system. The critical mass is thus calculated
        as: 

            m_crit = R_h**(7/3) * ( 4 * pi * mass_per_star / 3 * Sigma_0 * tau * G**(1/2) )**(2/3),
        
        where R_h is the half-mass radius, mass_per_star is the typical mass of a single star, tau is the age of the cluster
        (which we are approximating as the evolution of the cluster since the begining of the simulation) and G the 
        gravitational constant. Sigma_0 is the effective cross-section defined as:

            Sigma_0 = 16 * sqrt(pi) * stellar_radius**2 * (1 + theta)
        
        With stellar_radius the typical radius of a star in the cluster, and theta the Safronov number. Safronov number is 
        computed internally using safronov_num() function.
    ________________________________________________________________________________________________________________________
    """
    # Convert to arrays for validation
    hm_radius      = np.atleast_1d(hm_radius)
    mass_per_star  = np.atleast_1d(mass_per_star)
    cluster_age    = np.atleast_1d(cluster_age)
    v_disp         = np.atleast_1d(v_disp)
    stellar_radius = np.atleast_1d(stellar_radius)
    
    # Define units
    mass_unit = u.solMass
    
    # Input validation
    if np.any(hm_radius <= 0):
        raise ValueError("Half-mass radius must be positive")
    if np.any(mass_per_star <= 0):
        raise ValueError("Stellar mass must be positive")
    if np.any(cluster_age <= 0):
        raise ValueError("Cluster age must be positive")
    if np.any(v_disp <= 0):
        raise ValueError("Velocity dispersion must be positive")
    if np.any(stellar_radius <= 0):
        raise ValueError("Stellar radius must be positive")

    # Compute Safronov number (dimensionless)
    theta, theta_unit = safronov_num(mass_per_star, v_disp, stellar_radius)

    # Convert stellar radius from solar radii to pc
    stellar_radius_pc = (stellar_radius * u.solRad).to(u.pc).value

    # Compute Sigma_0 (effective cross-section): Sigma_0 = 16 * sqrt(pi) * R_star^2 * (1 + theta)
    Sigma_0 = 16 * np.sqrt(np.pi) * stellar_radius_pc**2 * (1 + theta)

    # Convert G to appropriate units: pc^3 / (solar_mass * Myr^2)
    G_value = c.G.to(u.pc**3 / (mass_unit * u.Myr**2)).value

    # Compute the factor: (4 * pi * mass_per_star) / (3 * Sigma_0 * tau * sqrt(G))
    factor = (4 * np.pi * mass_per_star) / (3 * Sigma_0 * cluster_age * np.sqrt(G_value))

    # Compute critical mass: m_crit = R_h^(7/3) * factor^(2/3)
    m_crit = hm_radius**(7/3) * factor**(2/3)

    # Validate that the result is positive
    if np.any(m_crit <= 0):
        raise ValueError("Critical mass calculation resulted in a non-positive value")

    return m_crit, mass_unit

# Aproximation of the stellar radius --------------------------------------------------------------------------------------#
def mean_stellar_radius(stellar_mass : Union[float, np.ndarray]) -> Tuple[np.ndarray, u.Unit]:
    """
    ________________________________________________________________________________________________________________________
    Calculate the mean stellar radius based on the mean stellar mass using a mass-radius relation. 
    (Unit conversion is handled by Astropy)
    ________________________________________________________________________________________________________________________
    Parameters:
        stellar_mass (Union[float, np.ndarray]) [Msun] : Stellar mass. Mandatory.
    ________________________________________________________________________________________________________________________
    Returns:
        stellar_radius, radius_unit (Tuple[np.ndarray, u.Unit]) [Rsun]: Mean stellar radius corresponding to the given 
                                                                       stellar mass. Returns array with values and astropy 
                                                                       length unit.
    ________________________________________________________________________________________________________________________
    Notes:
        The mass-radius relation used here is a simplified approximation for main-sequence stars:

                R* = 1.01*(M/Msun)^0.724 Rsun, for 0.1 < M < 10 Msun (Demircan & Kahraman 1991, ZAMS models fit)
                R* = 1.6*(M/Msun)^0.47 Rsun,   for 10. < M < 50 Msun (Bond et al. 1984)
                R* = 0.85*(M/Msun)^0.67 Rsun,  for M >=50 Msun       (Demircan & Kahraman 1991, Empirical fit)

        This relation provides a reasonable estimate of the stellar radius based on mass for a wide range of stellar types.
    ________________________________________________________________________________________________________________________
    """
    # Convert to array for validation
    stellar_mass = np.atleast_1d(stellar_mass)

    # Define units
    mass_unit   = u.solMass
    radius_unit = u.solRad

    # Input validation
    if np.any(stellar_mass <= 0):
        raise ValueError("Stellar mass must be positive")

    # Initialize array for stellar radius
    stellar_radius = np.zeros_like(stellar_mass)

    # Low mass range: 0.1 < M < 10 Msun (Demircan & Kahraman 1991)
    mask_low_mass = (stellar_mass >= 0.1) & (stellar_mass < 10.0)
    stellar_radius[mask_low_mass] = 1.01 * (stellar_mass[mask_low_mass])**0.724
    
    # Medium mass range: 10 < M < 50 Msun (Bond et al. 1984)
    mask_medium_mass = (stellar_mass >= 10.0) & (stellar_mass < 50.0)
    stellar_radius[mask_medium_mass] = 1.6 * (stellar_mass[mask_medium_mass])**0.47
    
    # High mass range: M >= 50 Msun (Demircan & Kahraman 1991)
    mask_high_mass = stellar_mass >= 50.0
    stellar_radius[mask_high_mass] = 0.85 * (stellar_mass[mask_high_mass])**0.67
    
    # Handle edge case: masses below 0.1 Msun (extrapolate with low-mass relation)
    mask_very_low_mass = stellar_mass < 0.1
    if np.any(mask_very_low_mass):
        stellar_radius[mask_very_low_mass] = 1.01 * (stellar_mass[mask_very_low_mass])**0.724
    
    # Validate that result is reasonable (should be positive)
    if np.any(stellar_radius <= 0):
        raise ValueError("Stellar radius calculation resulted in non-positive value")
    
    return stellar_radius, radius_unit

# Computation of the Galactocentric Radius --------------------------------------------------------------------------------#
def galactocentric_radius(tildal_radius: Union[float, np.ndarray], 
                          total_mass   : Union[float, np.ndarray],
                          v_circ       : float = 220.0) -> Tuple[np.ndarray, u.Unit]:  # v_circ in km/s
    """
    ________________________________________________________________________________________________________________________
    Calculate the Galactocentric radius  (Unit conversion is handled by Astropy)
    ________________________________________________________________________________________________________________________
    Parameters:
        tildal_radius (Union[float, np.ndarray]) [pc]   : Tidal radius of the cluster.
        total_mass    (Union[float, np.ndarray]) [Msun] : Total mass of the cluster.
        v_circ        (float)                    [km/s] : Circular velocity at the solar radius. Default is 220 km/s.
    ________________________________________________________________________________________________________________________
    Returns:
        r_gal, r_unit (Tuple[np.ndarray, u.Unit]) [pc]: Galactocentric radius calculated as 
    ________________________________________________________________________________________________________________________
    Notes:
        The Galactocentric radius is computed by inverting the classical tidal-radius relation assuming a spherical host 
        galaxy with a flat rotation curve (isothermal sphere potential) and a circular orbit:

                Rgc = sqrt((3.0*(v_circ**2)*(tildal_radius**3))/(G*total_mass))
        
        This expression assumes:
            - a flat rotation curve (M_host(R) ∝ R),
            - a circular orbit,
            - the tidal radius corresponds to the instantaneous (or pericentric) orbit,
            - the cluster mass is fully enclosed within the tidal radius.

        Deviations from these assumptions (e.g., eccentric orbits, time-dependent tides, or non-spherical potentials) may 
        lead to systematic uncertainties in the estimated Galactocentric radius.

        Unit conversion is handled internally using Astropy
    ________________________________________________________________________________________________________________________
    """
    # Convert to arrays for validation
    tildal_radius = np.atleast_1d(tildal_radius)
    total_mass    = np.atleast_1d(total_mass)
    
    # Define units
    radius_unit = u.pc
    mass_unit   = u.solMass
    v_circ_unit = u.km / u.s
    
    # Input validation
    if np.any(tildal_radius <= 0):
        raise ValueError("Tidal radius must be positive")
    if np.any(total_mass <= 0):
        raise ValueError("Total mass must be positive")
    if v_circ <= 0:
        raise ValueError("Circular velocity must be positive")
    
    # Convert G to appropriate units: pc^3 / (solar_mass * (km/s)^2)
    G_value = c.G.to(radius_unit**3 / (mass_unit * v_circ_unit**2)).value
    
    # Calculate Galactocentric radius: Rgc = sqrt((3.0 * v_circ^2 * r_tidal^3) / (G * M_total))
    r_gal = np.sqrt((3.0 * v_circ**2 * tildal_radius**3) / (G_value * total_mass))
    
    # Validate that result is reasonable (should be positive)
    if np.any(r_gal <= 0):
        raise ValueError("Galactocentric radius calculation resulted in non-positive value")
    
    return r_gal, radius_unit

#--------------------------------------------------------------------------------------------------------------------------#
