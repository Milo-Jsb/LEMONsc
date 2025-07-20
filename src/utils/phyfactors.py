# Modules -----------------------------------------------------------------------------------------------------------------#
import numpy         as np
import astropy.units as u

# External functions and utilities ----------------------------------------------------------------------------------------#
from typing  import Optional
from astropy import constants as c

# Computation of crossing times -------------------------------------------------------------------------------------------#
def crossing_time(hm_radius: float, v_disp: Optional[float] = None, mtot: Optional[float] = None) ->  u.Quantity:
    """
    ________________________________________________________________________________________________________________________
    Calculate the crossing time of a star cluster. (Unit conversion is handled by Astropy)
    ________________________________________________________________________________________________________________________
    Parameters:
        hm_radius (float)           [pc]   : Half-mass radius of the star cluster. Mandatory.
        v_disp    (Optional[float]) [km/s] : Velocity dispersion of the star cluster. Optional.
        mtot      (Optional[float]) [Msun] : Total mass of the star cluster. Mandatory if v_disp is None.
    ________________________________________________________________________________________________________________________
    Returns:
        t_cross   (unit.Quantity)   [Myr]  : Crossing time of the star cluster (time that takes a star to travel across the 
                                             whole system under the influence of the gravitational field of the cluster).
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
    # Input validation
    if hm_radius <= 0:
        raise ValueError("Half-mass radius must be positive")
    
    # Convert inputs to astropy quantities
    hm_radius = hm_radius * u.pc

    # Handle lack of velocity dispersion
    if v_disp is None:
        if mtot is None:
            raise ValueError("Total mass must be provided when velocity dispersion is not given")
        if mtot <= 0:
            raise ValueError("Total mass must be positive")
        
        # Calculate velocity dispersion using the virial theorem approximation
        G    = c.G
        mtot = mtot * u.solMass

        # Calculate velocity dispersion: v_disp = sqrt((3*G*Mtot)/(5*Rh))
        v_disp = np.sqrt((3 * G * mtot) / (5 * hm_radius))
        
        # Convert to km/s for consistency with input format
        v_disp = v_disp.to(u.km / u.s)
    else:
        if v_disp <= 0:
            raise ValueError("Velocity dispersion must be positive")
        
        # Convert input velocity dispersion to astropy units
        v_disp = v_disp * u.km / u.s
    
    # Calculate crossing time
    t_cross =  hm_radius / v_disp
    
    # Convert to Myr as specified in the docstring
    t_cross = t_cross.to(u.Myr)
    
    # Assert that the result is in Myr units
    assert t_cross.unit == u.Myr, f"Crossing time must be in Myr, got {t_cross.unit}"

    # Return as Quantity
    return t_cross

# Computation of number density within half-mass radius ------------------------------------------------------------------#
def rho_at_rh(n_stars: int, hm_radius: float) -> u.Quantity:
    """
    ________________________________________________________________________________________________________________________
    Calculate the number density within the half-mass radius of a star cluster. (Unit conversion is handled by Astropy)
    ________________________________________________________________________________________________________________________
    Parameters:
        n_stars   (int)             [#]    : Number of stars in the cluster. Mandatory.
        hm_radius (float)           [pc]   : Half-mass radius of the star cluster. Mandatory.
    ________________________________________________________________________________________________________________________
    Returns:
        n_density (unit.Quantity)   [pc^-3]: Number density within the half-mass radius of the star cluster, assuming a 
                                             uniform distribution of stars across the cluster volume.
    ________________________________________________________________________________________________________________________
    Notes:
        Formula assumes a uniform distribution of stars across the cluster volume. The number density is approximated as:
        
                n_density = (3 * n_stars) / (8 * pi * hm_radius^3)
        
        This approximation is valid for a homogeneous sphere and provides a reasonable estimate of the stellar number 
        density within the half-mass radius of the cluster.
    ________________________________________________________________________________________________________________________
    """
    # Input validation
    if n_stars <= 0:
        raise ValueError("Number of stars in the cluster must be greater than zero")
    if hm_radius <= 0:
        raise ValueError("Half-mass radius must be positive")
    
    # Convert inputs to astropy quantities
    hm_radius = hm_radius * u.pc
    
    # Calculate number density within half-mass radius
    n_density = (3 * n_stars) / (8 * np.pi * hm_radius**3)
    
    # Assert that the result is in pc^-3 units
    assert n_density.unit == u.pc**(-3), f"Number density must be in pc^-3, got {n_density.unit}"
    
    # Validate that result is reasonable (should be positive)
    if n_density.value <= 0:
        raise ValueError("Number density calculation resulted in non-positive value")
    
    # Return as Quantity
    return n_density

# Computation of relaxation time ------------------------------------------------------------------------------------------#
def relaxation_time(n_stars: int, hm_radius: float, 
                    v_disp : Optional[float] = None, 
                    mtot   : Optional[float] = None) ->  u.Quantity:
    """
    ________________________________________________________________________________________________________________________
    Calculate the relaxation time of a star cluster. (Unit conversion is handled by Astropy)
    ________________________________________________________________________________________________________________________
    Parameters:
        n_stars   (int)             [#]    : Number of stars in the cluster. Mandatory
        hm_radius (float)           [pc]   : Half-mass radius of the star cluster. Mandatory.
        v_disp    (Optional[float]) [km/s] : Velocity dispersion of the star cluster. Optional.
        mtot      (Optional[float]) [Msun] : Total mass of the star cluster. Mandatory if v_disp is None.
    ________________________________________________________________________________________________________________________
    Returns:
        t_relax   (unit.Quantity)   [Myr]  : Relaxation time of the star cluster (time required for stellar encounters to 
                                             significantly alter the velocities of stars and drive the system toward 
                                             energy equipartition).
    ________________________________________________________________________________________________________________________
    Notes:
        Formula from Binney & Tremaine, Galactic Dynamics, Second Edition (Eq 1.38). The relaxation time is given by:
        
                t_relax = 0.1 * (N * t_cross) / ln(N)
        
        Where N is the number of stars, t_cross is the crossing time. Crossing time is computed internally using 
        phyfactors.crossing_time() function.
    ________________________________________________________________________________________________________________________
    """
    # Input validation
    if n_stars <= 0:
        raise ValueError("Number of stars in the cluster must be greater than zero")
    if n_stars < 10:
            raise ValueError("Relaxation time calculation requires at least 10 stars for meaningful results")
    
    # Compute crossing time (returns Quantity in Myr)
    t_cross = crossing_time(hm_radius, v_disp, mtot)

    # Compute relaxation time
    t_relax = 0.1* ((n_stars*t_cross)/np.log(n_stars))

    # Assert that the result is in Myr units
    assert t_relax.unit == u.Myr, f"Relaxation time must be in Myr, got {t_relax.unit}"

    # Return Quantity
    return t_relax

# Computation of core collapse time ---------------------------------------------------------------------------------------#
def core_collapse_time(m_mean: float, m_max: float, n_stars: int, hm_radius: float, 
                    v_disp : Optional[float] = None, 
                    mtot   : Optional[float] = None) ->  u.Quantity:
    """
    ________________________________________________________________________________________________________________________
    Calculate the core collapse time of a star cluster. (Unit conversion is handled by Astropy)
    ________________________________________________________________________________________________________________________
    Parameters:
        m_mean    (float)           [Msun] : Mean mass per star. Mandatory.
        m_max     (float)           [Msun] : Maximum stellar mass (considering the initial range of stellar masses). 
                                             Mandatory.
        n_stars   (int)             [#]    : Number of stars in the cluster. Mandatory
        hm_radius (float)           [pc]   : Half-mass radius of the star cluster. Mandatory.
        v_disp    (Optional[float]) [km/s] : Velocity dispersion of the star cluster. Optional.
        mtot      (Optional[float]) [Msun] : Total mass of the star cluster. Mandatory if v_disp is None.
    ________________________________________________________________________________________________________________________
    Returns:
        t_cc      (unit.Quantity)   [Myr]  : Core collapse time of the star cluster, which accounts for the presence of
                                             massive stars. The time is scaled by the ratio of the mean stellar mass to the 
                                             maximum stellar mass, reflecting the accelerated collapse driven by mass 
                                             segregation.
    ________________________________________________________________________________________________________________________
    Notes:
        Formula from Portegies & McMillan (2002) accounting for mass segregation: 
        
                t_cc = 3.3 * (m_mean / m_max) * t_relax
        
        Where t_relax is the relaxation time. Relaxation time is computed internally using phyfactors.relaxation_time() 
        function.
    ________________________________________________________________________________________________________________________
    """
     # Input validation
    if m_mean <= 0:
        raise ValueError("mean mass per star must be greater than zero")
    if m_max <= 0:
        raise ValueError("max mass must be greater than zero")
    if m_mean > m_max:
        raise ValueError("mean mass cannot exceed maximum mass")
    
    # Convert inputs to astropy quantities
    m_mean = m_mean * u.solMass
    m_max  = m_max  * u.solMass

    # Compute relaxation time
    t_relax = relaxation_time(n_stars, hm_radius, v_disp, mtot)

    # Compute core collapse time
    t_cc    = 3.3 * (m_mean / m_max) * t_relax

    # Assert that the result is in Myr units
    assert t_cc.unit == u.Myr, f"Core collapse time must be in Myr, got {t_cc.unit}"
    
    # Return Quantity
    return t_cc

# Computation of safronov number ------------------------------------------------------------------------------------------#
def safronov_num(mass_per_star: float, v_disp: float, stellar_radius: float = 1.0) -> float:
    """
    ________________________________________________________________________________________________________________________
    Calculate the Safronov number of a star cluster. (Unit conversion is handled by Astropy)
    ________________________________________________________________________________________________________________________
    Parameters:
        mass_per_star   (float)     [Msun] : Typical stellar mass per star in the cluster. Mandatory.
        v_disp          (float)     [km/s] : Velocity dispersion of the star cluster. Mandatory.
        stellar_radius  (float)     [Rsun] : Stellar radius. Default is 1.0 Rsun.
    ________________________________________________________________________________________________________________________
    Returns:
        theta           (float)     [#]    : Safronov number, a dimensionless parameter that measures the importance of 
                                             gravitational focusing in stellar encounters. Values >> 1 indicate strong 
                                             gravitational focusing, while values << 1 indicate weak focusing.
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
    # Input validation
    if mass_per_star <= 0:
        raise ValueError("Stellar mass must be positive")
    if v_disp <= 0:
        raise ValueError("Velocity dispersion must be positive")
    if stellar_radius <= 0:
        raise ValueError("Stellar radius must be positive")
    
    # Convert inputs to astropy quantities with proper units
    mass_per_star  = mass_per_star  * u.solMass
    v_disp         = v_disp         * u.km / u.s
    stellar_radius = stellar_radius * u.solRad
    
    # Calculate Safronov number using astropy units
    solar_mass         = 1.0   * u.solMass
    solar_radius       = 1.0   * u.solRad
    reference_velocity = 100.0 * u.km / u.s
    
    theta = 9.54 * (mass_per_star / solar_mass) * (solar_radius / stellar_radius) * (reference_velocity / v_disp)**2
    
    # Ensure the result is dimensionless
    theta = theta.decompose()
    
    # Assert that the result is dimensionless
    assert theta.unit.is_unity(), f"Safronov number must be dimensionless, got {theta.unit}"
    
    # Validate that result is reasonable (should be positive)
    if theta.value <= 0:
        raise ValueError("Safronov number calculation resulted in non-positive value")
    
    # Return only the value number
    return theta.value

# Computation of collision time -------------------------------------------------------------------------------------------#
def collision_time(hm_radius: float, n_stars: int, mass_per_star: float, v_disp: float, 
                   stellar_radius : float = 1.0) -> u.Quantity:
    """
    ________________________________________________________________________________________________________________________
    Calculate the collision time of a star cluster. (Unit conversion is handled by Astropy)
    ________________________________________________________________________________________________________________________
    Parameters:
        hm_radius      (float) [pc]   : Half-mass radius of the star cluster. Mandatory.
        n_stars        (int)   [#]    : Number of stars in the cluster. Mandatory.
        mass_per_star  (float) [Msun] : Typical stellar mass per star in the cluster. Mandatory.
        v_disp         (float) [km/s] : Velocity dispersion of the star cluster. Mandatory.
        stellar_radius (float) [Rsun] : Stellar radius. Default is 1.0 Rsun.
    ________________________________________________________________________________________________________________________
    Returns:
        t_coll (unit.Quantity) [Myr] : Collision time of the star cluster (average time between stellar collisions, 
                                       accounting for gravitational focusing effects).
    ________________________________________________________________________________________________________________________
    Notes:
        Formula from Binney & Tremaine, Galactic Dynamics, Second Edition (Eq. 7.195). The expression includes 
        gravitational focusing. The collision time is given by:

                t_coll = (16 * sqrt(pi) * n * v_disp * stellar_radius**2 * (1+ theta))**(-1)
        
        With n the number density within the half mass radius, v_disp the velocity dispersion, stellar_radius the stellar 
        radius, and theta the Safronov number. the number density is being approximated as:

                n = (3 * n_stars ) / (8 * pi * hm_radius**3),

        Assuming a uniform distribution across the cluster. Safronov number is computed internally using 
        phyfactors.safronov_num() function.
    ________________________________________________________________________________________________________________________
    """
    # Input validation
    if hm_radius <= 0:
        raise ValueError("Half-mass radius must be positive")
    if n_stars <= 0:
        raise ValueError("Number of stars must be positive")
    if mass_per_star <= 0:
        raise ValueError("Stellar mass must be positive")
    if v_disp <= 0:
        raise ValueError("Velocity dispersion must be positive")
    if stellar_radius <= 0:
        raise ValueError("Stellar radius must be positive")
    
    # Convert inputs to astropy quantities
    hm_radius      = hm_radius      * u.pc
    v_disp         = v_disp         * u.km / u.s
    stellar_radius = stellar_radius * u.solRad
    
    # Calculate number density within half-mass radius
    number_density = (3 * n_stars) / (8 * np.pi * hm_radius**3)
    
    # Calculate Safronov number (this function returns a dimensionless value)
    theta = safronov_num(mass_per_star, v_disp.value, stellar_radius.value)
    
    # Calculate collision time
    t_coll = 1.0 / (16 * np.sqrt(np.pi) * number_density * v_disp * stellar_radius.to(u.pc)**2 * (1 + theta))
    
    # Convert to Myr as specified in the docstring
    t_coll = t_coll.to(u.Myr)
    
    # Assert that the result is in Myr units
    assert t_coll.unit == u.Myr, f"Collision time must be in Myr, got {t_coll.unit}"
    
    # Validate that result is reasonable (should be positive)
    if t_coll.value <= 0:
        raise ValueError("Collision time calculation resulted in non-positive value")
    
    # Return quantity
    return t_coll

# Critical Mass -----------------------------------------------------------------------------------------------------------#
def critical_mass(hm_radius: float, mass_per_star: float, cluster_age: float, 
                  v_disp: float, stellar_radius: float = 1.0) -> u.Quantity:
    """
    ________________________________________________________________________________________________________________________
    Calculate the critical mass of a star cluster. (Unit conversion is handled by Astropy)
    ________________________________________________________________________________________________________________________
    Parameters:
    
        hm_radius       (float) [pc]    : Half-mass radius of the star cluster. Mandatory.
        mass_per_star   (float) [Msun]  : Typical stellar mass per star in the cluster. Mandatory.
        cluster_age     (float) [Myr]   : Age of the cluster (or time since beginning of the simulation). Mandatory.
        v_disp          (float) [km/s]  : Velocity dispersion of the cluster. Mandatory.
        stellar_radius  (float) [Rsun]  : Typical stellar radius. Default is 1.0 Rsun.
        
    ________________________________________________________________________________________________________________________
    Returns:
        m_crit (unit.Quantity) [Msun] : Critical mass of the star cluster, beyond which the system may undergo runaway
                                        collisions or core collapse.
    ________________________________________________________________________________________________________________________
    Notes:
        Formula from Vergara et al. (2023), also present in Vergara et al (2024) However we do not take into account the
        term related to an external potential due to the presence of gas in the system. The critical mass is thus calculated
        as: 

            m_crit = R_h**(7/3) * ( 4 * pi * mass_per_star / 3 * Sigma_0 * tau * G**(1/2) )**(2/3),
        
        where R_h is the half-mass radius, mass_per_star is the typical mass of a single star, tau is the age of the cluster
        (which we are approximating as the evolution of the cluster since the begining of the simulation) and G the 
        gravitational constant. Sigma_0 is the the effective cross-section defined as:

            Sigma_0 = 16 * sqrt(pi) * stellar_radius**2 * (1 + theta)
        
        With stellar_radius the typical radius of a star in the cluster, and theta the Safronov number. Safronov number is 
        computed internally using phyfactors.safronov_num() function.
    ________________________________________________________________________________________________________________________
    """
     # Input validation
    if hm_radius <= 0:
        raise ValueError("Half-mass radius must be positive")
    if mass_per_star <= 0:
        raise ValueError("Stellar mass must be positive")
    if cluster_age <= 0:
        raise ValueError("Cluster age must be positive")
    if v_disp <= 0:
        raise ValueError("Velocity dispersion must be positive")
    if stellar_radius <= 0:
        raise ValueError("Stellar radius must be positive")

    # Convert inputs to astropy quantities
    R_h     = hm_radius      * u.pc
    m_star  = mass_per_star  * u.solMass
    tau     = cluster_age    * u.Myr
    v_disp  = v_disp         * u.km / u.s
    R_star  = stellar_radius * u.solRad

    # Compute Safronov number (dimensionless)
    theta = safronov_num(m_star.value, v_disp.value, R_star.value)

    # Compute Sigma_0 (effective cross-section)
    Sigma_0 = 16 * np.sqrt(np.pi) * R_star.to(u.pc)**2 * (1 + theta)

    # Compute the factor inside big "()"
    factor = (4 * np.pi * m_star) / (3 * Sigma_0 * tau * np.sqrt(c.G))

    # Compute m_crit using the full expression
    m_crit = R_h**(7/3) * factor**(2/3)

    # Convert result to solar masses
    m_crit = m_crit.to(u.solMass)

    # Assert output is in solar masses
    assert m_crit.unit == u.solMass, f"Critical mass must be in solar masses, got {m_crit.unit}"

    # Validate that the result is positive
    if m_crit.value <= 0:
        raise ValueError("Critical mass calculation resulted in a non-positive value")

    # Return quantity
    return m_crit
#--------------------------------------------------------------------------------------------------------------------------#
