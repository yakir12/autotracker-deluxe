import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import sys
import matplotlib.transforms as transforms
import matplotlib.colors as colors
import pandas as pd

from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Ellipse
from math import floor as fl
from random import randint
from scipy.interpolate import CubicSpline, interp1d, UnivariateSpline
from scipy.integrate import cumtrapz
from scipy.signal import find_peaks, butter, lfilter, hilbert, correlate, spectrogram
from scipy.fftpack import rfft, irfft, rfftfreq, fft, ifft, fftfreq
from matplotlib.widgets import Slider, Button
from scipy import stats, signal



# analyze
def analyze(dir, r_full, tau_full, tau_split, fps):

    len_scale = 1

    n_split = 0
    if len(tau_split):
        n_split = len(tau_split)

    tau_split = np.append(0,tau_split)
    tau_split = np.append(tau_split,tau_full[-1])

    r_s_us = []
    r_data_save = []
    t_save = []

    r_ref = np.array([0,0])

    for i in range(n_split+1): # I think this is for each track

        # tau_full sliced from min(a) to max(b)
        # a = np.where(tau_full >= tau_split[i])
        # b = np.where(tau_full < tau_split[i+1])
        # Segments out the time window in which the current track occurs;
        # form an array of timestamps.
        tau = tau_full[np.min(np.where(tau_full >= tau_split[i])):np.max(np.where(tau_full < tau_split[i+1]))]
        
        # Smoothing spline interpolation

        # Line below does: t = np.arange(duration, 1/fps)
        # Gives you a frame index from the time window 'tau'
        t = np.arange(0, 
                      tau[-1] - tau[0], 
                      1/fps)
        
        # np.where used in this way returns a weird tuple (array,)
        # I then have no clue how this gets interpreted. I presume it just gets
        # treated as an np arraylike: np.squeeze(result)[()] so it reduces
        # to a scalar or to an array as required.

        # I think this does the same thing as the tau setting above.
        tau_sort_arg = np.where((tau_full >= tau_split[i]) & (tau_full < tau_split[i+1]))

        # Zero the timeseries window extracted above so that time starts at the
        # start of the window. Don't know why sorting is included. Is the 
        # timeseries not monotonic?
        tau_sorted = np.sort(tau - tau[0])

        # Removes singleton dimensions and (I'm guessing) makes this a 1D array 
        # where each element corresponds to a timestamp in tau_sort_arg
        r = np.squeeze(r_full[:,tau_sort_arg])

        # I think these two lines simply zero the tracks. I think r has one 
        # row per coordinate
        # [[x1,x2,...];
        #  [y1,y2,...]]
        r_ref = r[:,0]
        r = np.transpose(np.transpose(r) - r_ref) 

        # Rename r to r_data for some reason presumably
        r_data = r

        # Store the r and t data in out-of-loop datastructures (presumably
        # for later storage)
        r_data_save.append(np.array(r_data))
        t_save.append(np.array(t))

        # Insert 0 at the beginning of the sorted time series, then compute
        # the difference between each consecutive pair of elements (derivative?)
        tau_diff = np.ediff1d(np.insert(tau_sorted,0,0))

        # Time window, presumably for smoothing. Analyse frames in one second
        # chunks
        window = int(1*fps)

        # interp1d(x, y, kind='linear'); linear interpolation of x coordinate 
        # with respect to time. But only where the time between entries is 
        # small enough? (<=1.5/fps).
        # Remember these are objects which are called for a given value
        x_lin = interp1d(tau_sorted[np.where(tau_diff <= 1.5/fps)], 
                         r[0, np.where(tau_diff <= 1.5/fps)], 
                         kind='linear')
        y_lin = interp1d(tau_sorted[np.where(tau_diff <= 1.5/fps)], 
                         r[1,np.where(tau_diff <= 1.5/fps)], 
                         kind='linear')


        #
        # I think w_x and w_y below are identical computations on the x and y
        # dimensions. 
        # These are used as weights for spline fitting
        #

        # w_x = np.sqrt(a) 
        # a = np.reciprocal(b) (1/x for x in b)
        # b = const + np.convolve(u, v**2,mode='same') - np.convolve(u, v, mode='same')**2 (does const simply avoid div by zero?)
        # u = np.ones(FPS)/FPS (FPS is scalar; 1/FPS of length int(FPS) ~ 24)
        # v = np.ediff1d(p, to_begin=0, to_end=0) (derivative bookended with 0s)
        # p = np.squeeze(x_lin(t)) Interpolated x w.r.t. time. Not sure why squeeze used.

        w_x = np.sqrt(
            np.reciprocal(
                1e-6 +\
                np.convolve(
                    np.ones(window)/window, 
                    np.ediff1d(
                        np.squeeze(x_lin(t)), 
                        to_end=0, 
                        to_begin=0)**2, 
                        mode='same') -\
                (np.convolve(
                    np.ones(window)/window,
                    np.ediff1d(
                        np.squeeze(x_lin(t)), 
                        to_end=0,
                        to_begin=0), 
                    mode='same'))**2 
                )
            )
        w_x = w_x/np.max(w_x)

        
        w_y = np.sqrt(np.reciprocal(1e-6 + np.convolve(np.ones(window)/window, np.ediff1d(np.squeeze(y_lin(t)),to_end=0,to_begin=0)**2, mode='same') - (np.convolve(np.ones(window)/window, np.ediff1d(np.squeeze(y_lin(t)),to_end=0,to_begin=0), mode='same'))**2 ))
        w_y = w_y/np.max(w_y)

        # Repack interpolated (x,y) into a new array r_lin. Commented.
        # r_lin = np.squeeze(np.array([x_lin(t),y_lin(t)]))


        # Taking x as a function of time, fit a spline to the raw data to smooth
        # it out. I don't understand the weight computation here and will probably
        # need to ask. Same process is used for y
        x_us = UnivariateSpline(tau_sorted[np.where(tau_diff <= 1.5/fps)], 
                                r[0,np.where(tau_diff <= 1.5/fps)], 
                                w=w_x[np.where(tau_diff <= 1.5/fps)], 
                                k=3, 
                                s=0.0005*len(np.where(tau_diff <= 1.5/fps)))
        y_us = UnivariateSpline(tau_sorted[np.where(tau_diff <= 1.5/fps)], r[1,np.where(tau_diff <= 1.5/fps)], w=w_y[np.where(tau_diff <= 1.5/fps)], k=3, s=0.0005*len(np.where(tau_diff <= 1.5/fps)) )

        # plt.plot(w_x)
        # print(y_us.get_residual())

        # Take first and second derivatives of x and y splines
        vx_us = x_us.derivative(n=1)
        vy_us = y_us.derivative(n=1)
        ax_us = x_us.derivative(n=2)
        ay_us = y_us.derivative(n=2)

        # Each (x,y) on the spline for each point in time (Unused?)
        r_us = np.squeeze(np.array([x_us(t),y_us(t)]))

        # (x',y') for each point in time
        v_us = np.squeeze(np.array([vx_us(t),vy_us(t)]))

        #(x'', y'') for each point in time. (Unused?)
        a_us = np.squeeze(np.array([ax_us(t),ay_us(t)]))

        # Below is named 'speed', not sure how. norm will compute the vector
        # norm along axis 0 (columns). I'm not entirely sure of the form of
        # v_us at this point. I think it's [[x1, x2, x3], [y1, y2, y3]]
        # (for the derivatives of x and y) which means you're 
        
        # 'speed' computes the magnitude of the vector which describes the
        # derivative of the beetle's position, for each point in time. 
        # Note this reduces speed in the x,y plane to a scalar. 

        # Arclength parametrization 
        speed = np.linalg.norm(v_us, axis=0)

        # Cumulative integration of speed w.r.t. time, with a starting value 
        # of 0. Total distance travelled?
        arc = cumtrapz(speed, t, initial=0)

        # Fitting a spline to a spline... Computing smoothed values of x and y 
        # w.r.t. distance travelled. Note, have now figured out why my smoothing
        # wasn't working the way I wanted. I should use this instead of
        # the modern equivalent of interp1d.
        x_s_us = UnivariateSpline(arc, x_us(t), s=0)
        y_s_us = UnivariateSpline(arc, y_us(t), s=0)

        # Derivatives of smoothed x and y w.r.t. distance travelled.
        # vx_s_us = x_s_us.derivative(n=1)
        # vy_s_us = y_s_us.derivative(n=1)
        # ax_s_us = x_s_us.derivative(n=2)
        # ay_s_us = y_s_us.derivative(n=2)

        # Fitting more splines. Smoothing x and y w.r.t. 'tau_sorted'
        # then again, computing the smoothed x and y w.r.t. distance travelled.
        x_us_temp = UnivariateSpline(tau_sorted[np.where(tau_diff <= 1.5/fps)], r_data[0,np.where(tau_diff <= 1.5/fps)], w=w_x[np.where(tau_diff <= 1.5/fps)], k=3, s=0.0005*len(np.where(tau_diff <= 1.5/fps)) )
        y_us_temp = UnivariateSpline(tau_sorted[np.where(tau_diff <= 1.5/fps)], r_data[1,np.where(tau_diff <= 1.5/fps)], w=w_y[np.where(tau_diff <= 1.5/fps)], k=3, s=0.0005*len(np.where(tau_diff <= 1.5/fps)) )
        x_s_us_temp = UnivariateSpline(arc, x_us_temp(t), s=0)
        y_s_us_temp = UnivariateSpline(arc, y_us_temp(t), s=0)

        # Derivatives have been overwritten by 'temp' values?
        vx_s_us = x_s_us_temp.derivative(n=1)
        vy_s_us = y_s_us_temp.derivative(n=1)
        ax_s_us = x_s_us_temp.derivative(n=2)
        ay_s_us = y_s_us_temp.derivative(n=2)

        # Creating a range from the total distance travelled 
        s = np.arange(0, arc[-1]-arc[0], 0.005/len_scale)

        # Speed smoothed w.r.t. distance travelled, indexed using linear speed
        # range and then normed? (Unused?)
        v_s_us = np.sqrt(vx_s_us(s)**2+vy_s_us(s)**2)

        # Append the smoothed track to the 'global' data structure (r_s_us)
        # which stores the tracks. 
        r_s_us.append(np.array([x_s_us(s),y_s_us(s)]))
        # r_s_us = np.squeeze(np.array([x_s_us(s),y_s_us(s)]))
        # r_s_us = np.transpose((np.transpose(r_s_us) - r_ref)/len_scale)

        # Multiplying x velocity by y accelaration, then subtracting the 
        # product of y velocity and x accelaration
        kappa_s = np.multiply(vx_s_us(s),ay_s_us(s)) - np.multiply(vy_s_us(s),ax_s_us(s))
        kappa_s_hist = np.histogram(kappa_s)

        scale_min = 1
        scale_max = 50

        # Correlation of kappa_s with itself but not sure why the mean is 
        # subtracted.
        autocorr_kappa = np.correlate(kappa_s - np.mean(kappa_s), 
                                      kappa_s - np.mean(kappa_s),
                                      'full') / (kappa_s.shape[0]*np.std(kappa_s)**2)


        
        # Again multiplying x velocity by y acceleration and subtracting the
        # product of y velocity and x acceleration. Then dividing by the
        # square sum of velocity of x and y (to the power of 1.5?).

        # I don't know the significance of the vx*ay - vy*ax calculation. It
        # happens a lot here. Nor do I know the significance of the where
        # condition here.

        # Note that np.multiply and np.divide are elementwise operations.
        # The calculation is done for each point in time.

        # Variables of interest
        kappa = np.divide(np.multiply(vx_us(t),ay_us(t)) - np.multiply(vy_us(t), ax_us(t)), 
                          (vx_us(t)**2 + vy_us(t)**2)**1.5 , 
                          out=np.zeros_like(t), 
                          where=(vx_us(t)**2 + vy_us(t)**2)**0.5 >= 1e-4)

        # Same as kappa with different power in the denomenator and different
        # related where condition. Same is true for u_N. I actually don't
        # see an obvious difference with u_T and u_N. Ah, u_T is multiplying
        # the dimension velocity with its own acceleration. u_N works 
        # across dimensions
        u_T = np.divide( np.multiply(vx_us(t), ax_us(t)) + np.multiply(vy_us(t),ay_us(t)), 
                        (vx_us(t)**2 + vy_us(t)**2)**0.5 , 
                        out=np.zeros_like(t), 
                        where= (vx_us(t)**2 + vy_us(t)**2)**0.5 >=1e-4 )
        u_N = np.divide(np.multiply(vx_us(t), ay_us(t)) - np.multiply(vy_us(t),ax_us(t)), 
                        (vx_us(t)**2 + vy_us(t)**2)**0.5, 
                        out=np.zeros_like(t), 
                        where= (vx_us(t)**2 + vy_us(t)**2)**0.5 >=1e-4 )

        # Same cross multiplication for velocity and accelerationm divided by the
        # squared sum of the velocities in each dimension (which is the 
        # squared norm of the velocity vector...). I don't know what this represents
        omega = np.divide(np.multiply(vx_us(t),ay_us(t)) - np.multiply(vy_us(t),ax_us(t)), 
                          (vx_us(t)**2 + vy_us(t)**2),
                          out=np.zeros_like(t), 
                          where= (vx_us(t)**2 + vy_us(t)**2)**0.5 >=1e-4 )
        
        # Integral of omega w.r.t. time
        theta = cumtrapz(omega, t, initial=0)

        # Omega and integral w.r.t. distance as opposed to time.
        omega_s = np.multiply(vx_s_us(s),ay_s_us(s)) - np.multiply(vy_s_us(s),ax_s_us(s))
        del_omega_s = np.ediff1d(np.insert(omega_s,0,0))
        theta_s = cumtrapz(omega_s, s, initial=0)

        # Changes in the cumulative integral of omega w.r.t. distance.
        del_theta_s = np.ediff1d(np.insert(theta_s,0,0))

        # Looking at the angle of del_theta_s
        del_theta_s = np.arctan2(np.sin(del_theta_s), np.cos(del_theta_s))
        del_theta_s_df = pd.DataFrame(del_theta_s)
        del_theta_s_ma = del_theta_s_df.rolling(window=50,center=True).mean().to_numpy()
        del_theta_s_mstd = del_theta_s_df.rolling(window=50,center=True).std().to_numpy()

        integral_theta_s = cumtrapz(theta_s, s, initial=theta_s[0])
        int_mean_theta_s = np.divide(integral_theta_s[1:], s[1:])

        tangent_x_s = np.cos(theta_s)
        tangent_y_s = np.sin(theta_s)

        tangent_corr = correlate(tangent_x_s, tangent_x_s,'full') + correlate(tangent_y_s, tangent_y_s,'full')
        ndata = np.concatenate((np.arange(1,tangent_x_s.shape[0]+1), np.arange(tangent_x_s.shape[0],1,-1)))
        tangent_corr = np.divide(tangent_corr,ndata)
        tangent_corr_envelope = np.abs(hilbert(tangent_corr))

        tangent_corr_cwt = signal.cwt(tangent_corr, signal.ricker, np.arange(scale_min, scale_max, 1, dtype=float))
        tangent_corr_cwt = np.flipud(tangent_corr_cwt)

        tangent_corr_mat = np.outer(tangent_x_s, tangent_x_s) + np.outer(tangent_y_s, tangent_y_s)

        n_window = tangent_corr_mat.shape[0]//2

        tangent_corr_timefreq = []
        for k in range(n_window-1):
            tangent_corr_timefreq.append(tangent_corr_mat[k][k:k+n_window-1])

        tangent_corr_timefreq = np.array(tangent_corr_timefreq)

        phi_s = np.arctan2(r_s_us[i][1,:],r_s_us[i][0,:])*180/np.pi

        #
        # Only r datastructures are plotted below; all of the additional stats
        # computed above aren't used. Distance is also plotted w.r.t. time. 
        # I think this is total distance travelled over the track and not total
        # displacement (i.e. this only lets you see if the beetle sat still 
        # for some time).
        #



    # KDE
    # xmin = np.min(r[0,:])
    # xmax = np.max(r[0,:])
    # ymin = np.min(r[1,:])
    # ymax = np.max(r[1,:])

    # X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    # positions = np.vstack([X.ravel(), Y.ravel()])
    #
    # values = np.vstack([r[0,:], r[1,:]])
    # kernel = stats.gaussian_kde(values)
    # rho = np.reshape(kernel(positions).T, X.shape)

    # pre_mode = [X.ravel()[np.argmax(rho_pre.ravel())], Y.ravel()[np.argmax(rho_pre.ravel())]]
    # pre_mode_scale = pre_mode/d_scale

    # r_seq = np.linalg.norm(np.transpose(np.transpose(r_s_us) - r_ref), axis=0)

    # Plots
    fig1, track = plt.subplots()
    fig1.tight_layout()
    track.grid(True, which='both')
    track.axhline(y=0, color='k')
    track.axvline(x=0, color='k')
    # track.set(xlabel="X",ylabel="Y")
    # track.set_autoscale_on(False)
    # track.set_aspect('equal')
    # track.axis([xmin,xmax,ymin,ymax])
    track.legend(loc='lower right', fontsize='large')

    fig2, (an1,an2) = plt.subplots(2,1)
    fig2.tight_layout()

    an1.grid(True, which='both')
    an1.axhline(y=0, color='k')
    an1.axvline(x=0, color='k')
    # track.set(xlabel="X",ylabel="Y")
    # an.set_autoscale_on(False)
    an2.grid(True, which='both')
    an2.axhline(y=0, color='k')
    an2.axvline(x=0, color='k')
    # track.set(xlabel="X",ylabel="Y")
    # an.set_autoscale_on(False)


    for i in range(n_split+1):
        track.scatter(r_data_save[i][0,:],r_data_save[i][1,:], s=1, c='r', zorder=3, label='data')
        track.scatter(r_s_us[i][0,0], r_s_us[i][1,0], s=5, c='k', marker='*', zorder=3)
        # track.scatter(r_data[0,np.min(np.where(arc-arc[0] >= s_start))],r_data[1,np.min(np.where(arc-arc[0] >= s_start))], s=100, c='k', marker='+', zorder=4, label='data')
        track.plot(r_s_us[i][0,:], r_s_us[i][1,:] , 'b', zorder=1, label='rolls')
        # track.quiver(r_s_us[0,:], r_s_us[1,:], vx_s_us(s), vy_s_us(s), zorder=3, headlength = 2, scale=50)
        # track.set(xlabel="$\kappa$",ylabel="$|v|$")
        # # track.scatter(np.abs(kappa_s[np.squeeze(np.where(s<s_drop))]), v_s_us[[np.squeeze(np.where(s<s_drop))]], s=100, c='r', marker='+')
        # track.scatter(np.abs(kappa), np.linalg.norm(v_us,axis=0), s=50, c='r', marker='.',zorder=3)

        np.savetxt(dir + 'analysis_data/' + 'roll_' + str(i) + '.csv', np.array(r_s_us[i]), delimiter=',')
        np.savetxt(dir + 'analysis_data/' + 't_' + str(i) + '.csv', np.array(t_save[i]), delimiter=',')
        np.savetxt(dir + 'analysis_data/' + 'raw_data_roll_' + str(i) + '.csv', np.array(r_data_save[i]), delimiter=',')

    an1.set(xlabel="time",ylabel="path length")
    an1.plot(t,arc)

    # an2.set(xlabel="path length",ylabel="straightness")
    # an2.plot(s,np.divide(r_seq,s))



    plt.show()
