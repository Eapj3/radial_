import numpy as np
import matplotlib.pyplot as plt
import orbit
import estimate
import time

# Simple radial velocities example
def RVs_example():
    """
    Example on obtaining radial velocities using the parameters of the star 
    HD 156846 and its planet HD 156846 b.
    """
    ts = np.linspace(3600., 4200., 1000)
    start_time = time.time()
    RVs = orbit.log_RVs(lnK = np.log(0.464),
                        lnT = np.log(359.51),
                        t0 = 3998.1,
                        w = 52.2,
                        lne = np.log(0.847),
                        lna = np.log(0.9930),
                        VZ = -68.54,
                        NT = 1000,
                        ts = ts)
    print('RV calculation took %.4f seconds' % (time.time()-start_time))

    plt.plot(ts, RVs)
    plt.xlabel('JD - 2450000.0 (days)')
    plt.ylabel('RV (km/s)')
    plt.show()

# Maximum likelihood estimation example
def ML_example():
    """
    Example of maximum likelihood estimation of the orbital parameters
    of HD 156846 b
    """
    # The true parameters of the orbit of HD 156846 b
    ts = np.linspace(3600., 4200., 100)
    VZ = -68.54
    NT = 1000
    K_true = 0.464
    T_true = 359.51
    t0_true = 3998.1
    w_true = 52.2
    e_true = 0.847
    a_true = 0.9930
    RVs = orbit.get_RVs(K = K_true,
                        T = T_true,
                        t0 = t0_true,
                        w = w_true,
                        e = e_true,
                        a = a_true,
                        VZ = VZ,
                        NT = NT,
                        ts = ts)
    # Introducing noise to the RVs
    RV_d = np.array([RVk + np.random.normal(loc=0., scale=0.08) for RVk in RVs])
    t_d = np.array([tk + np.random.normal(loc=0., scale=0.1) for tk in ts])
    RV_derr = np.array([0.03 + np.random.normal(loc=0.0, scale=0.005) \
                        for k in RVs])
    # Estimating the orbital parameters using the true parameters as guess
    guess = [K_true, T_true, t0_true, w_true, e_true, a_true], 
    bnds = ((0, 1),(300, 400),(3600, 4200),(0, 360),(0, 1),(0, 5))
    print('Starting maximum likelihood estimation.')
    start_time = time.time()
    params_ml = estimate.ml_orbit(t_d, RV_d, RV_derr,
                                  guess = guess,
                                  bnds = bnds,
                                  VZ = VZ, NT = NT)
    print('Orbital parameters estimation took %.4f seconds' % \
          (time.time()-start_time))
    print('K = %.3f, T = %.2f, t0 = %.1f, w = %.1f, e = %.3f, a = %.4f' % \
          (params_ml[0], params_ml[1], params_ml[2], params_ml[3], 
           params_ml[4], params_ml[5]))
    RV_est = orbit.get_RVs(K = params_ml[0],
                           T = params_ml[1],
                           t0 = params_ml[2],
                           w = params_ml[3],
                           e = params_ml[4],
                           a = params_ml[5],
                           VZ = VZ,
                           NT = NT,
                           ts = ts)
    # Plotting the results
    plt.errorbar(t_d, RV_d, fmt='.', yerr = RV_derr, label="Simulated data")
    plt.plot(ts, RVs, label="True orbit")
    plt.plot(ts, RV_est, label="Estimated orbit")
    plt.plot()
    plt.xlabel('JD - 2450000.0 (days)')
    plt.ylabel('RV (km/s)')
    plt.legend(numpoints=1)
    plt.show()
