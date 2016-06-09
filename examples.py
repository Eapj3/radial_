import numpy as np
import matplotlib.pyplot as plt
import orbit
import estimate
import time


# Simple radial velocities example
def rvs_example():
    """
    Example on obtaining radial velocities using the parameters of the star 
    HD 156846 and its planet HD 156846 b.
    """
    ts = np.linspace(3600., 4200., 1000)
    start_time = time.time()
    rvs = orbit.log_rvs(log_k=np.log(0.464),
                        log_period=np.log(359.51),
                        t0=3998.1,
                        w=52.2,
                        log_e=np.log(0.847),
                        vz=-68.54,
                        nt=1000,
                        ts=ts)
    print('RV calculation took %.4f seconds' % (time.time()-start_time))

    plt.plot(ts, rvs)
    plt.xlabel('JD - 2450000.0 (days)')
    plt.ylabel('RV (km/s)')
    plt.show()


# Maximum likelihood estimation example
def ml_example():
    """
    Example of maximum likelihood estimation of the orbital parameters
    of HD 156846 b
    """
    # The true parameters of the orbit of HD 156846 b
    ts = np.linspace(3600., 4200., 100)
    vz = -68.54
    nt = 1000
    k_true = 0.464
    period_true = 359.51
    t0_true = 3998.1
    w_true = 52.2
    e_true = 0.847
    rvs = orbit.get_rvs(k=k_true,
                        period=period_true,
                        t0=t0_true,
                        w=w_true,
                        e=e_true,
                        vz=vz,
                        nt=nt,
                        ts=ts)
    # Introducing noise to the RVs
    rv_d = np.array([rvk + np.random.normal(loc=0., scale=0.08) for rvk in rvs])
    t_d = np.array([tk + np.random.normal(loc=0., scale=0.1) for tk in ts])
    rv_derr = np.array([0.03 + np.random.normal(loc=0.0, scale=0.005)
                        for k in rvs])
    # Estimating the orbital parameters using the true parameters as guess
    guess = [k_true, period_true, t0_true, w_true, e_true],
    bnds = ((0, 1), (300, 400), (3600, 4200), (0, 360), (0, 1))
    print('Starting maximum likelihood estimation.')
    start_time = time.time()
    params_ml = estimate.ml_orbit(t_d, rv_d, rv_derr,
                                  guess=guess,
                                  bnds=bnds,
                                  vz=vz, nt=nt)
    print('Orbital parameters estimation took %.4f seconds' %
          (time.time()-start_time))
    print('K = %.3f, T = %.2f, t0 = %.1f, w = %.1f, e = %.3f' %
          (params_ml[0], params_ml[1], params_ml[2], params_ml[3], 
           params_ml[4]))
    rv_est = orbit.get_rvs(k=params_ml[0],
                           period=params_ml[1],
                           t0=params_ml[2],
                           w=params_ml[3],
                           e=params_ml[4],
                           vz=vz,
                           nt=nt,
                           ts=ts)
    # Plotting the results
    plt.errorbar(t_d, rv_d, fmt='.', yerr=rv_derr, label="Simulated data")
    plt.plot(ts, rvs, label="True orbit")
    plt.plot(ts, rv_est, label="Estimated orbit")
    plt.plot()
    plt.xlabel('JD - 2450000.0 (days)')
    plt.ylabel('RV (km/s)')
    plt.legend(numpoints=1)
    plt.show()


ml_example()