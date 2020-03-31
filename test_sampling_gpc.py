import multiprocessing
from pathlib import Path
import os
import time

import mkl
mkl.set_num_threads(8)

import numpy as np

import matplotlib
matplotlib.use("SVG")
import matplotlib.pyplot as plt

import util
from test_common import Tester
from BO_common import random_hypersphere
from GPy_wrapper import GPyWrapper as GP
from GPy_wrapper import GPyWrapper_Classifier as GPC
import GP_util

import config
import config_model

import random_walk as rw
from pygor_fixvol import PygorRewire

def save(save_dir, vols_poff_all, u_all, EI_all, r_all, d_all, vols_poff_axes_all, poff_all, detected_all, logger, extra_measure, obj_val, time_all, time_acq_opt_all, dropped=None):
    # save all data
    vols_poff_all.dump(str(save_dir / 'vols_poff.npy'))
    u_all.dump(str(save_dir / 'unit_vector.npy'))
    r_all.dump(str(save_dir / 'dist_origin.npy'))
    d_all.dump(str(save_dir / 'dist_surface.npy'))
    vols_poff_axes_all.dump(str(save_dir / 'vols_poff_axes.npy'))
    poff_all.dump(str(save_dir / 'poff_allgates.npy'))
    np.array(EI_all).dump(str(save_dir / 'EI.npy'))
    np.save(str(save_dir / 'detected.npy'), detected_all)

    # logs for all line scan from the orign to the boundary
    vals_all = [item['val'] for item in logger]
    vols_all = [item['vols'] for item in logger]
    pinchoff_all = [item['pinchoff_idx'] for item in logger]

    np.save(str(save_dir / 'raw_vals.npy'), vals_all)
    np.save(str(save_dir / 'raw_vols.npy'), vols_all)
    np.save(str(save_dir / 'raw_pinchoff_idx.npy'), pinchoff_all)

    np.save(str(save_dir / 'extra_measure'), extra_measure)

    np.save(str(save_dir / 'time_meas'), time_all)
    np.save(str(save_dir / 'time_acq'), time_acq_opt_all)

    obj_val.dump(str(save_dir / 'obj_val.npy'))

    if dropped is not None:
        np.save(str(save_dir / 'dropped.npy'), dropped)

def main():
    conf_name = 'config2'
    if conf_name == 'config1':
        conf_func = config.config1
        ip = "http://129.67.86.107:8000/RPC2"
        save_dir = Path('./save_Dominic_ABL_group_optparam_run2')
        settletime, settletime_big = 0.01, 0.03 # settletime_big is for shuttling, 'None' disables shuttling
        # ['c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10']
        origin = -100.
        threshold_low = 0.0 # for pinchoff detector
        d_r = 10 # length of one step
    elif conf_name == 'config2':
        # ['c3', 'c4', 'c8', 'c10', 'c11', 'c12', 'c16']
        conf_func = config.config2
        ip = 'http://129.67.85.38:8000/RPC2'
        save_dir = Path('./save_Basel2_group')
        settletime, settletime_big = 0.02, 0.02 # settletime_big is for shuttling, 'None' disables shuttling
        origin = -100.
        threshold_low = 2.e-11
        d_r = 10 # length of one step
    elif conf_name == 'dummy':
        import pygor_dummy
        box_dim = 5
        box_a = -1000. * np.ones(box_dim)
        box_b = 1000. * np.ones(box_dim)
        shape = pygor_dummy.Box(ndim=box_dim, a=box_a, b=box_b)
        th_leak = np.array([-500., -400., -300., 0., 0.])
        shape = pygor_dummy.Leakage(shape, th_leak)
        save_dir = Path('./save_dummy')
        origin = -100.
        threshold_low = 0.2
        d_r = 10 # length of one step
    else:
        raise ValueError('Unsupported setup')
    save_dir.mkdir(exist_ok=True)

    if conf_name != 'dummy':
        pg, active_gate_idxs, lb_short, ub_short, max_current, min_current, set_big_jump, set_small_jump = config.setup_pygor(conf_func, ip, settletime, settletime_big)
    else:
        pg, active_gate_idxs, lb_short, ub_short, max_current, min_current, set_big_jump, set_small_jump = config.setup_pygor_dummy(shape)
    threshold_high = 0.8 * max_current
    num_active_gates = len(active_gate_idxs)

    # gate names are for nothing
    active_gate_names = ["c{}".format(i+1) for i in active_gate_idxs]
    print(active_gate_names)

    new_wires = [[1, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 1],
         [0, 0, 1, 0, 0, 1, 0],
         [0, 0, 0, 1, 1, 0, 0]]
    pg = PygorRewire(pg, new_wires)
    lb_short = lb_short[:4]
    ub_short = ub_short[:4]
    num_active_gates = len(new_wires)

    # choose the origin
    if np.isscalar(origin):
        origin = origin*np.ones(num_active_gates)
    else:
        if len(origin) != num_active_gates:
            raise ValueError('Wrong array shape of the origin.')
    origin.dump(str(save_dir / 'origin.npy'))

    
    # important algorithm parameters
    #threshold_low = 0.2 * (max_current - min_current) + min_current
    step_back = 100 # important param for measuring d
    len_after_pinchoff=50
    num_samples = 2000 # number of initial samples (by Latin hypercube design)

    detector_pinchoff = util.PinchoffDetectorThreshold(threshold_low) # pichoff detector
    detector_conducting = util.ConductingDetectorThreshold(threshold_high) # reverse direction
    # create a Callable object, input: unit_vector, output: distance between the boundary and the origin
    tester = Tester(pg, lb_short, ub_short, detector_pinchoff, d_r=d_r, len_after_pinchoff=len_after_pinchoff, logging=True, detector_conducting=detector_conducting, set_big_jump = set_big_jump, set_small_jump = set_small_jump)

    ###
    # Set a problem and a model
    ###
    do_extra_meas = lambda vols, th: config_model.do_extra_meas(pg, vols, th)
    #do_extra_meas = None

    ###
    # Gaussian process for r
    ###
    l_prior_mean = 0.4 * np.ones(num_active_gates)
    l_prior_var = 0.1*0.1 * np.ones(num_active_gates)
    r_min, r_max =  0.0, np.sqrt(num_active_gates)* 2000.
    v_prior_mean = ((r_max-r_min)/4.0)**2
    #v_prior_var = v_prior_mean**2
    #noise_var_r = np.square(d_r/2.0)

    gp_r = GP_util.create_GP(num_active_gates, 'Matern52', v_prior_mean, l_prior_mean, (r_max-r_min)/2.0)
    GP_util.set_GP_prior(gp_r, l_prior_mean, l_prior_var, None, None) # do not set prior for kernel var
    GP_util.fix_hyperparams(gp_r, False, True)

    ###
    # Gaussian process classifiers for predicting each probability component
    ###
    l_prior_mean =  500. * np.ones(num_active_gates)
    l_prior_var = 100.**2 * np.ones(num_active_gates)
    v_prior_mean =  50.
    v_prior_var = 20.**2
    gpc_dict = dict()
    gpc_dict['valid'] = GP_util.create_GP(num_active_gates, 'Matern52', lengthscale=l_prior_mean, const_kernel=True, GP=GPC)
    gpc_dict['peak'] = GP_util.create_GP(num_active_gates, 'Matern52', lengthscale=l_prior_mean, const_kernel=True, GP=GPC) # when a point is valid
    gpc_dict['goodscore'] = GP_util.create_GP(num_active_gates, 'Matern52', lengthscale=l_prior_mean, const_kernel=True, GP=GPC) # when a point is valid and has peaks

    GP_util.set_GP_prior(gpc_dict['valid'], l_prior_mean, l_prior_var, v_prior_mean, v_prior_var) # do not set prior for kernel var
    GP_util.set_GP_prior(gpc_dict['peak'], l_prior_mean, l_prior_var, v_prior_mean, v_prior_var) # do not set prior for kernel var
    GP_util.set_GP_prior(gpc_dict['goodscore'], l_prior_mean, l_prior_var, v_prior_mean, v_prior_var) # do not set prior for kernel var

    do_random_meas(num_active_gates, num_samples, tester, step_back, origin, lb_short, ub_short, gp_r, gpc_dict, do_extra_meas=do_extra_meas, save_dir=save_dir)
    print('Time for random sampling: ', time.time()-tic)

def update_gpc(gpc_dict, points_extended, poff_extended, pks_all, lowres_score, M):
    # All points for valid/invalid
    poff_extended = np.array(poff_extended)
    gpc_dict['valid'].create_model(points_extended, poff_extended.astype(np.float))

    # Only valid points for peaks/no peak
    assert len(pks_all) == len(lowres_score)
    num_scored = len(pks_all)
    is_poff = poff_extended[:num_scored]
    points_poff = points_extended[:num_scored][is_poff]
    pks_poff = np.array(pks_all)[is_poff]
    if len(np.unique(pks_poff)) < 2:
        return
    gpc_dict['peak'].create_model(points_poff, pks_poff.astype(np.float))
    #print('Training data, peak')
    #print(points_poff)
    #print(pks_poff)

    return

    # Only peak points for goodscore/not
    points_peaks = points_poff[pks_poff]
    lowres_score_peaks = lowres_score[is_poff][pks_poff]
    gt_threshold = lowres_score_peaks > M

    if len(np.unique(gt_threshold)) < 2:
        return
    gpc_dict['goodscore'].create_model(points_peaks, gt_threshold.astype(np.float))

def optimize_gpc(gpc_dict):
    for gpc in gpc_dict.values():
        if gpc.model is not None:
            gpc.optimize()

def predict_probs(points, gpc_dict):
    p1 = gpc_dict['valid'].predict_prob(points)[:,0]
    probs = [p1]

    if gpc_dict['peak'].model is not None:
        p2 = gpc_dict['peak'].predict_prob(points)[:,0]
        probs += [p2]

    if gpc_dict['goodscore'].model is not None:
        p3 = gpc_dict['goodscore'].predict_prob(points)[:,0]
        probs += [p3]

    total_prob = np.prod(probs, axis=0)
    log_total_prob = np.sum(np.log(probs), axis=0)
    return total_prob, log_total_prob, probs

def choose_next(points_candidate, points_observed, gpc_dict, d_tooclose = 100.):
    points_observed = np.array(points_observed)
    if len(points_candidate) == 0: # No cadidate points
        return None, None, None

    # Exclude samples that are too close to observed points
    tooclose = np.any(
            np.all(np.fabs(points_candidate[:,np.newaxis,:] - points_observed[np.newaxis,...]) <= d_tooclose, axis=2),
            axis=1)
    nottooclose = np.logical_not(tooclose)

    if np.sum(nottooclose) == 0: # All points are too close to observed points
        return None, None, None

    points_reduced = points_candidate[nottooclose]
    prob = predict_probs(points_reduced, gpc_dict)[0]
    p = prob / np.sum(prob)
    idx = np.random.choice(len(points_reduced), p=p)
    point_best =  points_reduced[idx]
    return point_best

def do_random_meas(num_active_gates, num_samples, tester, step_back, origin, lb_box, ub_box, gp_r, gpc_dict, do_extra_meas=None, save_dir=None):

    vols_poff_all = list()
    vols_poff_axes_all = list()
    d_all = list()
    poff_all = list()
    detected_all = list()
    time_all = list()
    u_all = list()
    r_all = list()
    extra_meas_all = list()
    time_removed_all = list()
    ub_history = list()
    changeindex_history = list()
    path_all = list()

    axes = list(range(num_active_gates))

    hardub = np.zeros(origin.size) # Hardbound of ub_samples
    ub_samples = origin.copy() # Random samples should be below this point
    origin_to_ub = True

    least_num_GP = 10 # minimum number of iterations for GP inference
    num_dvec = 30 # the number of iterations to collect d vector
    min_interval_GP_opt = 10 # minimum interval for GP inference
    steps_GP_inference = [least_num_GP]
    while steps_GP_inference[-1] < num_samples:
        prev = steps_GP_inference[-1]
        nextstep = np.maximum(min_interval_GP_opt, int(0.1*prev))
        steps_GP_inference += [prev + nextstep]
    th_lowres_lb = 0.001 # lower bound of the threshold for lowres score
    th_lowres = th_lowres_lb
    #th_lowres = 0.

    num_particles = 200 # number of particles
    samples = None
    point_selected = None
    boundary_points = []
    gp_availble = False
    while len(r_all) < num_samples:
        t = time.time()
        # Pick one surface point on the estimated surface
        if point_selected is not None:
            v = point_selected
            v_origin = v - origin
            u = v_origin / np.sqrt(np.sum(np.square(v_origin))) # voltage -> unit vector
        elif len(boundary_points) != 0:
            v = rw.pick_from_boundary_points(boundary_points) 
            v_origin = v - origin
            u = v_origin / np.sqrt(np.sum(np.square(v_origin))) # voltage -> unit vector
        else:
            print('WARNING: no boundary point is sampled')
            u = random_hypersphere(num_active_gates, 1)[0]
        print(u)

        # Start sampling
        sampling_on = False
        if gp_availble:
            # Initial samples that are inside of the hypersurface
            if samples is None:
                samples = rw.random_points_inside(num_active_gates, num_particles, gp_r, origin, lb_box, ub_samples)
            sampler = rw.create_sampler(gp_r, origin, lb_box, ub_samples, sigma=25)
            stopper = multiprocessing.Value('i', 0)
            listener, sender = multiprocessing.Pipe(duplex=False)
            sampler.reset(samples, max_steps=100000, stopper=stopper, result_sender=sender)
            sampler.start()
            sampling_on = True

        # Estimation of r
        if gp_availble:
            r_mean, r_var = gp_r.predict_f(u[np.newaxis,:], full_cov=False)
            r_est = np.maximum(r_mean - 2.0*np.sqrt(r_var), 0.0)
            r_est = r_est[0,0]
        else:
            r_est = None

        # Get measurements
        r, vols_pinchoff, found, t_firstjump = tester.get_r(u, origin=origin, r_est=r_est) # Measure the distance
        t1 = time.time() - t
        path_all.append((r_est or origin, vols_pinchoff))

        #if gp_availble:
        #    time.sleep(5)

        if len(r_all) >= num_dvec: axes=[] # do not measure d vector
        d_vec, poff_vec, meas_each_axis, vols_each_axis = tester.measure_dvec(vols_pinchoff+step_back, axes=axes)
        t2 = time.time() - t

        print('vols: ', vols_pinchoff, 'd_all: ', d_vec, 'poff: ', poff_vec)
        print('end_points:', vols_each_axis)

        # Store the measurement
        vols_poff_all.append(vols_pinchoff)
        vols_poff_axes_all.append(vols_each_axis)
        u_all.append(u)
        r_all.append(r)
        d_all.append(d_vec)
        poff_all.append(poff_vec)
        detected_all.append(found)

        # Extra measurement
        extra = meas_each_axis
        if found and do_extra_meas is not None:
            extra += do_extra_meas(vols_pinchoff, th_lowres)
        extra_meas_all.append(extra)
        t3 = time.time() - t

        # Stop sampling
        if sampling_on:
            stopper.value = 1
            counter, samples, boundary_points = listener.recv()
            sampler.join()
            print('Steps={} ({})'.format(counter, time.time()-t))

        # Compute hardbound
        change_hardbound, new_hardbound = util.compute_hardbound(poff_vec, found, vols_pinchoff, step_back, axes, hardub)
        if change_hardbound:
            hardub = new_hardbound
            outofbound = ub_samples > hardub
            if np.any(outofbound):
                ub_samples[outofbound] = hardub[outofbound]
            print('New upperbound: ', ub_samples)
            ub_history.append(ub_samples.copy())
            changeindex_history.append(len(r_all))
            if save_dir is not None:
                ub_samples.dump(str(save_dir / 'upperbound.npy'))
                np.array(ub_history).dump(str(save_dir / 'ub_history.npy'))
                np.array(changeindex_history).dump(str(save_dir / 'changeindex_history.npy'))

            if origin_to_ub:
                origin = ub_samples
                u_all, r_all = util.ur_from_vols_origin(vols_poff_all, origin)
                if save_dir is not None:
                    ub_samples.dump(str(save_dir / 'origin.npy'))
                    np.array(ub_history).dump(str(save_dir / 'origin_history.npy'))


        # Merge data of boundary points
        if len(r_all) >= min(least_num_GP, num_dvec):
            points_extended, poff_extended = util.merge_data(vols_poff_all, detected_all, vols_poff_axes_all[:min(num_dvec,len(r_all))], poff_all[:min(num_dvec,len(r_all))])

        # Update GP
        if len(r_all) >= least_num_GP:
            # Choose data for gp_r
            points_poff = points_extended[poff_extended]
            inside = np.all(points_poff < origin, axis=1)
            u_all_gp, r_all_gp = util.ur_from_vols_origin(points_poff[inside], origin, returntype='array')

            gp_r.create_model(u_all_gp, r_all_gp[:,np.newaxis], (tester.d_r/2)**2, noise_prior='fixed')
            gp_availble = True
            if len(r_all) in steps_GP_inference or (change_hardbound and origin_to_ub):
                gp_r.optimize(num_restarts=5, opt_messages=False, print_result=True)

        # Project old samples to inside of the new estimated surface
        #samples = np.minimum(samples, ub_samples[np.newaxis,:]-1.)
        if gp_availble and samples is not None:
            samples = rw.project_points_to_inside(samples, gp_r, origin, factor=0.99)
        #samples = np.maximum(samples, lb_box[np.newaxis,:]+1.)
            samples_outside = np.logical_or(np.any(samples>ub_samples), np.any(samples<lb_box))
            #print(lb_box, ub_samples)
            #print('samples outside: ', samples[samples_outside])
            samples[samples_outside] = ub_samples - 1.

        print('The number of collected samples: ', len(r_all))



        # Update the probabilities and choose the best sample
        if len(r_all) >= num_dvec:
            # Data preparation
            n = num_active_gates # short notation
            pks_all = np.array([len(ext[3+n]) >= 1 if len(ext)>=4+n else False for ext in extra_meas_all])
            lowres_score = np.array([ext[8+n] if len(ext)>=9+n else 0.0 for ext in extra_meas_all])
            th_lowres = np.maximum(th_lowres_lb, np.quantile(lowres_score, 0.85))
            
            #print(poff_extended[:len(pks_all)])
            print('peaks and lowres scores')
            print(pks_all)
            print(lowres_score)

            # Update GPC
            M = 1.e10
            update_gpc(gpc_dict, points_extended, poff_extended, pks_all, lowres_score, M)
            if len(r_all) in steps_GP_inference:
                optimize_gpc(gpc_dict)

            if len(boundary_points) > 0:
                # Calculate correct boundary_points
                points_candidate = rw.project_crosses_to_boundary(boundary_points, gp_r, origin)
                # Choose the best point in boundary_points
                point_selected = choose_next(points_candidate, vols_poff_all, gpc_dict, d_tooclose = 20.)
                print('Next point: ', point_selected)

                if point_selected is not None:
                    # Move the point to the safe bound, it is not going to happen, but just for safety
                    point_selected = np.maximum(point_selected, lb_box+1.)

            else:
                point_selected = None
            #point_selected = None

        t4 = time.time() - t

        print('Elapsed time: ', (t_firstjump, t1, t2, t3, t4))
        time_all.append((t_firstjump, t1, t2, t3, t4))
        if save_dir is not None:
            save(save_dir, np.array(vols_poff_all), np.array(u_all), [], np.array(r_all), np.array(d_all), np.array(vols_poff_axes_all), np.array(poff_all), detected_all, tester.logger, extra_meas_all, np.array([]), time_all, [])


    # make u_all 2dim array
    vols_poff_all = np.array(vols_poff_all)
    u_all = np.array(u_all)
    r_all = np.array(r_all)
    d_all = np.array(d_all)
    poff_all = np.array(poff_all)

    return vols_poff_all, u_all, r_all, d_all, poff_all, detected_all, time_all, time_removed_all, extra_meas_all, origin, vols_poff_axes_all


if __name__ == '__main__':
    main()
