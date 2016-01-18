"""
1) Generate posterior samples of full velocity vector for BHB stars
2) Compute Galactocentric cartesian velocities
3) In a shitty, spherical model for the Galaxy, compute the actions
"""

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
from astropy import log as logger
from astropy.io import ascii
import astropy.coordinates as coord
import astropy.units as u
import emcee
import numpy as np
import matplotlib.pyplot as pl
import gary.coordinates as gc
from gary.util import get_pool
from scipy.stats import norm

vcirc = 238 * u.km/u.s
vlsr = [-11.1, 12.24, 7.25] * u.km/u.s
gc_frame = coord.Galactocentric(galcen_distance=8.3*u.kpc)

# HACK: arbitrary choice...
sigma_halo = 200. # km/s

# HACK: hard set
# nwalkers = 64
# nmcmc_steps = 128
nwalkers = 16
nmcmc_steps = 16

# HACK: hard set
cache_nsteps = 16

def ln_posterior(p, coordinate, obs_vlos, err_vlos):
    pm_l,pm_b,vlos = p

    vgal = gc.vhel_to_gal(coordinate,
                          pm=(pm_l*u.mas/u.yr,pm_b*u.mas/u.yr),
                          rv=vlos*u.km/u.s,
                          vcirc=vcirc, vlsr=vlsr, galactocentric_frame=gc_frame)
    vtot = np.sqrt(np.sum(vgal**2)).to(u.km/u.s).value
    return norm.logpdf(vtot, loc=0., scale=sigma_halo) + norm.logpdf(vlos, loc=obs_vlos, scale=err_vlos)

def main(mpi=False):

    if not os.path.exists("data"):
        raise IOError("This must be run from the top level of the project.")

    np.random.seed(42)
    pool = get_pool(mpi=mpi)

    if not os.path.exists("output"):
        os.mkdir("output")

    # read the data
    tbl = ascii.read("data/xue08_bhb.txt")

    # set up the cache file
    cache_file = os.path.join("output", "samples.npy")
    cache_dtype = np.float64
    cache_shape = (len(tbl), cache_nsteps*nwalkers, 3)
    if not os.path.exists(cache_file):
        # make sure file exists
        cache = np.memmap(cache_file, dtype=cache_dtype, mode='w+', shape=cache_shape)
    cache = np.memmap(cache_file, dtype=cache_dtype, mode='r+', shape=cache_shape)

    # coordinate object and error in distance
    c = coord.SkyCoord(l=tbl['GLON'], b=tbl['GLAT'], distance=tbl['d'], frame='galactic')
    err_dist = u.Quantity(tbl['d']*0.1)

    # heliocentric line-of-sight velocity
    obs_vlos = u.Quantity(tbl['HRVel']).to(u.km/u.s).value
    err_vlos = u.Quantity(tbl['e_HRVel']).to(u.km/u.s).value

    sampler = None
    nstars = len(c)
    for i in range(nstars):
        logger.info("Star {}".format(i))
        if sampler is not None:
            sampler.reset()

        if np.any(cache[i] != 0.):
            logger.debug("Skipping -- data already cached...")
            continue

        logger.debug("Running MCMC sampling...")
        args = (c[i],obs_vlos[i],err_vlos)
        sampler = emcee.EnsembleSampler(nwalkers=nwalkers, dim=3, args=args,
                                        lnpostfn=ln_posterior, pool=pool)

        p0 = np.zeros((nwalkers,3))
        p0[:,0] = np.random.normal(0., 0.1, size=nwalkers) # mas/yr
        p0[:,1] = np.random.normal(0., 0.1, size=nwalkers) # mas/yr
        p0[:,2] = np.random.normal(obs_vlos[i], err_vlos[i]/10., size=nwalkers) # km/s
        _ = sampler.run_mcmc(p0, nmcmc_steps)

        cache[i] = np.vstack(sampler.chain[:,-16:])
        cache.flush()

if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        default=False, help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet",
                        default=False, help="Be quiet! (default = False)")
    parser.add_argument("--mpi", dest="mpi", action="store_true", default=False,
                        help="Run with MPI")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    main(mpi=args.mpi)
