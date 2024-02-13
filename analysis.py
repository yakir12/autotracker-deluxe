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

    for i in range(n_split+1):
        tau = tau_full[np.min(np.where(tau_full >= tau_split[i])):np.max(np.where(tau_full < tau_split[i+1]))]
        # Smoothing spline interpolation
        t = np.arange(0, tau[-1]-tau[0], 1/fps)
        tau_sort_arg = np.where((tau_full >= tau_split[i]) & (tau_full < tau_split[i+1]))
        tau_sorted = np.sort(tau - tau[0])
        r = np.squeeze(r_full[:,tau_sort_arg])

        # Reference
        r_ref = r[:,0]
        r = np.transpose(np.transpose(r) - r_ref)

        r_data = r
        r_data_save.append(np.array(r_data))
        t_save.append(np.array(t))

        tau_diff = np.ediff1d(np.insert(tau_sorted,0,0))

        window = int(1*fps)
        x_lin = interp1d(tau_sorted[np.where(tau_diff <= 1.5/fps)], r[0,np.where(tau_diff <= 1.5/fps)], kind='linear')
        y_lin = interp1d(tau_sorted[np.where(tau_diff <= 1.5/fps)], r[1,np.where(tau_diff <= 1.5/fps)], kind='linear')
        w_x = np.sqrt(np.reciprocal(1e-6 + np.convolve(np.ones(window)/window, np.ediff1d(np.squeeze(x_lin(t)),to_end=0,to_begin=0)**2, mode='same') - (np.convolve(np.ones(window)/window, np.ediff1d(np.squeeze(x_lin(t)),to_end=0,to_begin=0), mode='same'))**2 ))
        w_x = w_x/np.max(w_x)
        w_y = np.sqrt(np.reciprocal(1e-6 + np.convolve(np.ones(window)/window, np.ediff1d(np.squeeze(y_lin(t)),to_end=0,to_begin=0)**2, mode='same') - (np.convolve(np.ones(window)/window, np.ediff1d(np.squeeze(y_lin(t)),to_end=0,to_begin=0), mode='same'))**2 ))
        w_y = w_y/np.max(w_y)

        # r_lin = np.squeeze(np.array([x_lin(t),y_lin(t)]))

        x_us = UnivariateSpline(tau_sorted[np.where(tau_diff <= 1.5/fps)], r[0,np.where(tau_diff <= 1.5/fps)], w=w_x[np.where(tau_diff <= 1.5/fps)], k=3, s=0.0005*len(np.where(tau_diff <= 1.5/fps)) )
        y_us = UnivariateSpline(tau_sorted[np.where(tau_diff <= 1.5/fps)], r[1,np.where(tau_diff <= 1.5/fps)], w=w_y[np.where(tau_diff <= 1.5/fps)], k=3, s=0.0005*len(np.where(tau_diff <= 1.5/fps)) )

        # plt.plot(w_x)
        # print(y_us.get_residual())

        vx_us = x_us.derivative(n=1)
        vy_us = y_us.derivative(n=1)

        ax_us = x_us.derivative(n=2)
        ay_us = y_us.derivative(n=2)

        r_us = np.squeeze(np.array([x_us(t),y_us(t)]))
        v_us = np.squeeze(np.array([vx_us(t),vy_us(t)]))
        a_us = np.squeeze(np.array([ax_us(t),ay_us(t)]))

        # Arclength parametrization
        speed = np.linalg.norm(v_us,axis=0)
        arc = cumtrapz(speed,t, initial=0)

        x_s_us = UnivariateSpline(arc, x_us(t), s=0)
        y_s_us = UnivariateSpline(arc, y_us(t), s=0)

        # vx_s_us = x_s_us.derivative(n=1)
        # vy_s_us = y_s_us.derivative(n=1)
        # ax_s_us = x_s_us.derivative(n=2)
        # ay_s_us = y_s_us.derivative(n=2)

        x_us_temp = UnivariateSpline(tau_sorted[np.where(tau_diff <= 1.5/fps)], r_data[0,np.where(tau_diff <= 1.5/fps)], w=w_x[np.where(tau_diff <= 1.5/fps)], k=3, s=0.0005*len(np.where(tau_diff <= 1.5/fps)) )
        y_us_temp = UnivariateSpline(tau_sorted[np.where(tau_diff <= 1.5/fps)], r_data[1,np.where(tau_diff <= 1.5/fps)], w=w_y[np.where(tau_diff <= 1.5/fps)], k=3, s=0.0005*len(np.where(tau_diff <= 1.5/fps)) )
        x_s_us_temp = UnivariateSpline(arc, x_us_temp(t), s=0)
        y_s_us_temp = UnivariateSpline(arc, y_us_temp(t), s=0)
        vx_s_us = x_s_us_temp.derivative(n=1)
        vy_s_us = y_s_us_temp.derivative(n=1)
        ax_s_us = x_s_us_temp.derivative(n=2)
        ay_s_us = y_s_us_temp.derivative(n=2)

        s = np.arange(0, arc[-1]-arc[0], 0.005/len_scale)

        v_s_us = np.sqrt(vx_s_us(s)**2+vy_s_us(s)**2)

        r_s_us.append(np.array([x_s_us(s),y_s_us(s)]))
        # r_s_us = np.squeeze(np.array([x_s_us(s),y_s_us(s)]))
        # r_s_us = np.transpose((np.transpose(r_s_us) - r_ref)/len_scale)

        kappa_s = np.multiply(vx_s_us(s),ay_s_us(s)) - np.multiply(vy_s_us(s),ax_s_us(s))
        kappa_s_hist = np.histogram(kappa_s)

        scale_min = 1
        scale_max = 50
        autocorr_kappa = np.correlate(kappa_s -np.mean(kappa_s), kappa_s-np.mean(kappa_s),'full') / (kappa_s.shape[0]*np.std(kappa_s)**2)

        # Variables of interest
        kappa = np.divide( np.multiply(vx_us(t),ay_us(t)) - np.multiply(vy_us(t),ax_us(t)) , (vx_us(t)**2 + vy_us(t)**2)**1.5 , out=np.zeros_like(t), where= (vx_us(t)**2 + vy_us(t)**2)**0.5 >= 1e-4)

        u_T = np.divide( np.multiply(vx_us(t),ax_us(t)) + np.multiply(vy_us(t),ay_us(t)) , (vx_us(t)**2 + vy_us(t)**2)**0.5 , out=np.zeros_like(t), where= (vx_us(t)**2 + vy_us(t)**2)**0.5 >=1e-4 )
        u_N = np.divide( np.multiply(vx_us(t),ay_us(t)) - np.multiply(vy_us(t),ax_us(t)) , (vx_us(t)**2 + vy_us(t)**2)**0.5 , out=np.zeros_like(t), where= (vx_us(t)**2 + vy_us(t)**2)**0.5 >=1e-4 )

        omega = np.divide( np.multiply(vx_us(t),ay_us(t)) - np.multiply(vy_us(t),ax_us(t)) , (vx_us(t)**2 + vy_us(t)**2) , out=np.zeros_like(t), where= (vx_us(t)**2 + vy_us(t)**2)**0.5 >=1e-4 )
        theta = cumtrapz(omega,t, initial=0)

        omega_s = np.multiply(vx_s_us(s),ay_s_us(s)) - np.multiply(vy_s_us(s),ax_s_us(s))
        del_omega_s = np.ediff1d(np.insert(omega_s,0,0))
        theta_s = cumtrapz(omega_s, s, initial=0)

        del_theta_s = np.ediff1d(np.insert(theta_s,0,0))
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
