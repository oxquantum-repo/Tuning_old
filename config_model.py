# Define objective, feasibility function and corresponding GPs
import sys
import time
from functools import partial
import numpy as np
from scipy.stats import norm
from GPy_wrapper import GPyWrapper, GPyWrapper_MultiSeparate
from GPy_wrapper import GPyWrapper_Classifier as GPC
import GP_util
import BO_common
import Junk.dot_score
import Junk.dot_score_final as dot_score
from Pygor_new.scores import space_scoring
from Pygor_new.scores import count_scoring
from Pygor_new.measurement_functions import measurement_funcs as meas
from test_common import translate_window_inside_boundary
import score_driver
from Last_score import final_score_cls #Actually used
from pygor_fixvol import PygorRewire

def make_length_one(x):
    return x / np.sqrt(np.sum(np.square(x)))

class Normalize(object):
    def __call__(self, x ):
        return x / np.sqrt(np.sum(np.square(x)))
    def transform_withJ(self, x):
        ss = np.sum(np.square(x))
        rss = np.sqrt(ss)
        transformed = x/rss

        J = (np.eye(x.size) - np.outer(x,x)/ss)/rss
        return x, J

def L1_norm(arr, axis=None, keepdims=False):
    return np.sum(np.fabs(arr), axis=axis, keepdims=keepdims)

def L2_norm(arr, axis=None, keepdims=False):
    return np.sqrt(np.sum(np.square(arr), axis=axis, keepdims=keepdims))

def fill_nan(vals, ref_vals=None):
    vals = vals.copy()
    if vals.ndim == 1:
        if ref_vals is None:
            raise ValueError('Reference values are required to fill nan values for a vector.')
        nan_loc = np.isnan(vals)
        vals[nan_loc] = ref_vals[nan_loc]

    elif vals.ndim == 2:
        if ref_vals is None:
            ref_vals = np.nanmean(vals,axis=0)
        for i in range(vals.shape[1]):
            vals[np.isnan(vals[:,i]),i] = ref_vals[i]
    else:
        raise ValueError('Unsupported shape for filling nan values.')
    return vals

def objective_dtot_func(d_all, d_weights, detected_all=None, penalty_val=1000.):
    if d_all.ndim == 1:
        penalty = penalty_val if detected_all is False else 0.0
        return L2_norm(d_all*d_weights) + penalty
    else:
        if detected_all is None:
            penalty = 0.0
        else:
            assert len(detected_all) == len(d_all)
            penalty = [penalty_val if poff is False else 0.0 for poff in detected_all]
        return L2_norm(d_all*d_weights[np.newaxis,:], axis=-1) + penalty

def bool_pinchoff(poff, poff_eachaxis, poff_idxs=()):
    if len(poff_idxs) > 0:
        for poff_idx in poff_idxs:
            if poff_eachaxis.ndim == 2:
                poff = poff & poff_eachaxis[:,poff_idx]
            elif poff_eachaxis.ndim == 1:
                poff = poff & poff_eachaxis[poff_idx]
            else:
                raise ValueError('Pinchoff information should be 1 or 2 dimensional')
    return np.array(poff)

def EI_from_GP(gp, x, best_prev, std_noise=None, transform_x=None, min_problem=True, negative=False, grad=False):
    # assumes the shape of x is (num_dim)
    if transform_x is not None:
        if grad:
            x, J = transform_x.transform_withJ(x) # returns transformed value and Jacobian
        else:
            x = transform_x(x)
    x = x[np.newaxis,:]

    if grad:
        mean, std, dmdx, dsdx = gp.predict_withGradients(x)
    else:
        mean, var = gp.predict_f(x, full_cov=False)
        std = np.sqrt(var)
        dmdx = dsdx = None

    if std_noise is None:
        EI_val, dEIdx = BO_common.EI(best_prev, mean[0,0], std[0,0], min_problem, dmdx, dsdx)
    else:
        EI_val, dEIdx = BO_common.augmented_EI(best_prev, mean[0,0], std[0,0], std_noise, min_problem, dmdx, dsdx)
    #print(EI_val, dEIdx, J)
    #sys.exit(0)

    sign = -1.0 if negative else 1.0
    if grad:
        if transform_x:
            dEIdx = np.matmul(dEIdx, J)
        return sign * EI_val, sign * dEIdx.ravel()
    else:
        return sign * EI_val

def prod_except_itself(vals):
    result = np.zeros(vals.shape)
    for i in range(len(vals)):
        result[i] = np.prod(vals[:i]) * np.prod(vals[i+1:])
    return result

def feasibility_from_GPR(gp, x, th, transform_x=None, grad=False):
    # shape of x : (ndim)
    if transform_x is not None:
        if grad:
            x, J = transform_x.transform_withJ(x) # returns transformed value and Jacobian
        else:
            x = transform_x(x)
    x = x[np.newaxis,:] # (1 x ndim)

    if grad:
        mean, std, dmdx, dsdx = gp.predict_withGradients(x)
        #mean, std: (1 x outdim)
        #dmdx, dsdx: (1 x outdim x ndim)
    else:
        mean, var = gp.predict_f(x, full_cov=False)
        std = np.sqrt(var)
        dmdx = dsdx = None

    u = (th-mean)/std
    phi, Phi = norm.pdf(u), norm.cdf(u)
    prob = np.prod(Phi, axis=1)
    if grad:
        dPhidx = -(phi/std)[...,np.newaxis] * (u[...,np.newaxis]*dsdx + dmdx)
        if transform_x:
            dPhidx = np.matmul(dPhidx, J)
        # gradient of the product of Phi functions
        prod_except_i = prod_except_itself(Phi[0])
        dPdx = np.sum(dPhidx[0,...] * prod_except_i[:,np.newaxis], axis=0)
        return prob[0], dPdx
    else:
        return prob[0]

def feasibility_from_GPC(gpc, x, transform_x=None):
    # assumes the shape of x is (num_dim)
    if transform_x is not None:
        x = transform_x(x)[np.newaxis,:]
    else:
        x = x[np.newaxis,:]
    return gpc.predict_prob(x)[0,0]

class ProblemSpec(object):
    def evaluate_and_update(self, u_all, data):
        obj, meas_obj = self.objective(u_all, data)
        fea, meas_fea = self.feasibility(u_all, data)
        return obj, fea, meas_obj + meas+fea

    def objective(self, u_all, data):
        return 0, []

    def feasibility(self, u_all, data):
        return 1.0, [] # always feasible

    def pred_feasibility(self, uvec):
        return 1.0

    def acquisition_func(self, u_all):
        return 1.0

    def do_extra_meas(self, pg, vols):
        return [None]

class ProblemSpec_dtot(ProblemSpec):
    '''
    Objective: to reduce sqrt(sum of (weight*distance)**2) for each axis
    Feasiblity: pinchoff should be detected along u_vec, and along specified axes
    Remarks:
        - varaince hyperparameter for GP will not be inferred for the objective prediction, because it leads to overconfident. Instead, it uses expected min and max value of the objective functionto set the variance parameter
    '''
    def __init__(self, ndim, noise_var_obj, dvec_min, dvec_max, d_weights=1.0, poff_idxs=(), enable_feasibility=True, acq_criterion='EI'):
        self.ndim = ndim
        self.poff_idxs = poff_idxs
        self.enable_feasibility = enable_feasibility
        self.acq_criterion = acq_criterion
        self.noise_var_d = noise_var_obj

        if np.isscalar(d_weights):
            self.d_weights = d_weights*np.ones(ndim)
        elif len(d_weights) == ndim:
            self.d_weights = d_weights
        else:
            raise ValueError('Incorrect length of d_weights')

        ###
        # create GPs to predict the objective and the feasibility
        ###
        # for objective
        l_prior_mean = 0.2 * np.ones(ndim)
        l_prior_var = 0.1*0.1 * np.ones(ndim)
        obj_min, obj_max = objective_dtot_func(dvec_min, d_weights), objective_dtot_func(dvec_max, d_weights)
        v_prior_mean = ((obj_max-obj_min)/4.0)**2
        v_prior_var = v_prior_mean**2

        self.gp_d = GP_util.create_GP(ndim, 'Matern52', v_prior_mean, l_prior_mean, (obj_max-obj_min)/2.0)
        GP_util.set_GP_prior(self.gp_d, l_prior_mean, l_prior_var, None, None) # do not set prior for kernel var
        GP_util.fix_hyperparams(self.gp_d, False, True) # fix kernel variance

        # for feasibility
        if not enable_feasibility:
            return
        gpc_var = 5.**2
        gpc_lenthscale = 0.5
        l_prior_mean = 0.5 * np.ones(ndim)
        l_prior_var = 0.3*0.3 * np.ones(ndim)
        self.gpc = GPC()
        self.gpc.create_kernel(ndim, 'RBF', gpc_var, gpc_lenthscale)
        GP_util.set_GP_prior(self.gpc, l_prior_mean, l_prior_var, None, None) # do not set prior for kernel var

        #feasibility = lambda x: feasibility_from_GPC(self.gpc, x, transform_x=make_length_one)

    def set_data(self, u_all, data, do_opt=True):
        '''
        Set data for prediction, and optimize GPs
        Args:
            u_all: unit vector
            data: dictionary corresponding u_all
                'd_all': array (shape == u_all.shape), distance to a hypersurface for each axis
                'found': scalar (u_all.ndim==1) or 1D (u_all.ndim==2),  bool, True indicates pinchoff detected before touching a limit along the unit vector
                'poff_all': bool array (shape == u_all.shape), True indicates pinchoff detected along its axis before touching a limit
        Returns:
            Extra measurement used to evaluate the objective and feasibility
        '''
        self.u_all = u_all
        self.obj_data, meas_obj = self.objective(u_all, data) # data for computing objective value. For this model, obj_data == obj_val
        self.obj_val = self.obj_data # objective value should be a scalar value
        self.fea_data, meas_fea = self.feasibility(u_all, data)

        self.gp_d.create_model(u_all, self.obj_data, self.noise_var_d, noise_prior='fixed')
        self.gpc.create_model(u_all, self.fea_data.astype(float))

        if do_opt:
            self.gp_d.optimize(num_restarts=20, opt_messages=False, print_result=True)
            self.gpc.optimize()
        return meas_obj + meas_fea

    def objective(self, u_all, data):
        '''
        Evaluate the objective fuction
        Args:
            u_all: unit vector
            data: dictionary corresponding u_all
                'd_all': array (shape == u_all.shape), distance to a hypersurface for each axis
        Returns:
            real number, scalar (u_all.ndim==1) or 1D (u_all.ndim==2)
            empty list, meaning no extra measurement required
        '''
        d_all = data['d_all']
        return objective_dtot_func(d_all, self.d_weights), []

    def feasibility(self, u_all, data):
        '''
        Evaluate the feasibility fuction

        Args:
            u_all: 1D (ndim) or 2D (num_vectors x ndim) array
            data: dictionary corresponding u_all
                'found': scalar (u_all.ndim==1) or 1D (u_all.ndim==2),  bool, True indicates pinchoff detected before touching a limit along the unit vector
                'poff_all': bool array (shape == u_all.shape), True indicates pinchoff detected along its axis before touching a limit

        Returns:
            bool, scalar (u_all.ndim==1) or 1D (u_all.ndim==2)
            empty list, meaning no extra measurement required
        '''
        if not self.enable_feasibility:
            return 1.0, []
        found = data['found']
        poff_all = data['poff_all']
        return bool_pinchoff(found, poff_all, self.poff_idxs), []

    def pred_feasibility(self, uvec):
        return feasibility_from_GPC(self.gpc, uvec, make_length_one)

    def acquisition_func(self, noise_var_next=None):
        '''
        Returns:
            acqusition function
        '''
        if noise_var_next is None:
            noise_var = self.noise_var_d
        else:
            noise_var = noise_var_next

        if self.acq_criterion == 'EI':
            current_best_min = self.obj_data.min()
            func_EI = lambda x : -EI_from_GP(self.gp_d, x, current_best_min, transform_x=make_length_one) # negative sign for max -> min
        elif self.acq_criterion == 'augmented_EI':
            post_mean, var_post = self.gp_d.predict_f(u_all, full_cov=False)
            idx_min = np.argmin(post_mean)
            min_post_mean = post_mean[idx_min,0] # array shape of post_mean is num_points x 1
            func_EI = lambda x : -EI_from_GP(self.gp_d, x, min_post_mean, std_noise=np.sqrt(noise_var), transform_x=make_length_one) # negative sign for max -> min
        else:
            raise ValueError('Incorrect acqusition criterion.')

        if self.enable_feasibility:
            func_acq = lambda x: func_EI(x) * feasibility_from_GPC(self.gpc, x, make_length_one)
        else:
            func_acq = func_EI

        return func_acq

class ProblemSpec_score(ProblemSpec):
    def __init__(self, ndim, noise_var_obj, dvec_min, dvec_max, score_min, score_max, enable_feasibility=False, acq_criterion='EI', corner_th=300.):

        self.ndim = ndim
        self.enable_feasibility = enable_feasibility
        self.acq_criterion = acq_criterion
        self.noise_var_obj = noise_var_obj
        self.corner_th=corner_th

        ###
        # create GPs to predict the objective and the feasibility
        ###
        # for objective
        l_prior_mean = 0.1 * np.ones(ndim)
        l_prior_var = 0.1*0.1 * np.ones(ndim)
        v_prior_mean = ((score_max-score_min)/4.0)**2
        v_prior_var = v_prior_mean**2

        self.gp_score = GP_util.create_GP(ndim, 'Matern52', v_prior_mean, l_prior_mean, (score_max-score_min)/2.0)
        GP_util.set_GP_prior(self.gp_score, l_prior_mean, l_prior_var, None, None) # do not set prior for kernel var
        GP_util.fix_hyperparams(self.gp_score, False, True) # fix kernel variance

        # for feasibility
        if not enable_feasibility:
            return
        l_prior_mean = 0.2 * np.ones(ndim)
        l_prior_var = 0.1*0.1 * np.ones(ndim)
        d_min, d_max = 0., 2000.
        v_prior_mean = ((d_max-d_min)/4.0)**2
        v_prior_var = v_prior_mean**2

        self.gp_dall = GP_util.create_GP_multiout(ndim, ndim, 'Matern52', v_prior_mean, l_prior_mean, (d_max-d_min)/2.0, GP = GPyWrapper_MultiIndep)

        GP_util.set_GP_prior(self.gp_dall, l_prior_mean, l_prior_var, None, None) # do not set prior for kernel var
        GP_util.fix_hyperparams(self.gp_dall, False, True) # fix kernel variance

    def set_data(self, u_all, data, do_opt=True):
        '''
        Set data for prediction, and optimize GPs
        Args:
            u_all: unit vector
            data: dictionary corresponding u_all
        Returns:
            Extra measurement used to evaluate the objective and feasibility
        '''
        self.u_all = u_all
        self.obj_data, meas = self.objective(u_all, data) # data for computing objective value. For this model, obj_data == obj_val
        self.obj_val = self.obj_data # objective value should be a scalar value
        self.gp_score.create_model(u_all, self.obj_data, self.noise_var_obj, noise_prior='fixed')

        if self.enable_feasibility:
            self.fea_data, meas_fea = self.feasibility(u_all, data)
            self.gp_dall.create_model(u_all, self.fea_data)
            meas += meas_fea
        if do_opt:
            self.gp_score.optimize(num_restarts=20, opt_messages=False, print_result=True)
            if self.enable_feasibility:
                self.gp_dall.optimize(num_restarts=20, opt_messages=False, print_result=True)
        return meas


    def objective_single(self, u_all, ext_data):
        traces,coor,tracepoints,endpoints = ext_data[-1]
        s_traces = dot_score.smooth_trace(traces)
        return dot_score.execute_score(coor,s_traces[0:2],s_traces[2:4])

    def objective(self, u_all, data):
        '''
        Evaluate the objective fuction
        Args:
            u_all: unit vector
            data: dictionary corresponding u_all
        Returns:
            real number, scalar (u_all.ndim==1) or 1D (u_all.ndim==2)
            empty list, meaning no extra measurement required
        '''
        ext_data = data['ext_data']

        if u_all.ndim == 2:
            score = np.zeros(u_all.shape[0])
            for i, u_vec in enumerate(u_all):
                score[i] = self.objective_single(u_vec, ext_data[i])
        else:
            score = self.objective_single(u_all, ext_data)

        return score, []

    def feasibility(self, u_all, data):
        '''
        Evaluate the feasibility fuction

        Args:
            u_all: 1D (ndim) or 2D (num_vectors x ndim) array
            data: dictionary corresponding u_all
        Returns:
            bool, scalar (u_all.ndim==1) or 1D (u_all.ndim==2)
            empty list, meaning no extra measurement required
        '''
        if u_all.ndim == 1:
            num_vectors = 1
        else:
            num_vectors = u_all.shape[0]

        if not self.enable_feasibility:
            return np.ones(num_vectors), []
        return np.ones(num_vectors) , []

    def pred_feasibility(self, uvec):
        return 1.0

    def acquisition_func(self, noise_var_next=None):
        '''
        Returns:
            acqusition function
        '''
        if noise_var_next is None:
            noise_var = self.noise_var_obj
        else:
            noise_var = noise_var_next

        if self.acq_criterion == 'EI':
            current_best_max = self.obj_data.max()
            #func_EI = lambda x : -EI_from_GP(self.gp_score, x, current_best_max, transform_x=make_length_one, min_problem=False) # negative sign for max -> min
            func_EI = partial(EI_from_GP, gp=self.gp_score, best_prev=current_best_max, transform_x=make_length_one, min_problem=False, negative=True)
        elif self.acq_criterion == 'augmented_EI':
            post_mean, var_post = self.gp_d.predict_f(u_all, full_cov=False)
            idx_max = np.argmax(post_mean)
            max_post_mean = post_mean[idx_max,0] # array shape of post_mean is num_points x 1
            func_EI = lambda x : -EI_from_GP(self.gp_d, x, max_post_mean, std_noise=np.sqrt(noise_var), transform_x=make_length_one, min_problem=False) # negative sign for max -> min
        else:
            raise ValueError('Incorrect acqusition criterion.')

        if self.enable_feasibility:
            func_acq = lambda x: func_EI(x)
        else:
            func_acq = func_EI

        return func_acq

    def do_extra_meas(self, pg, vols):
        gates = ["c3","c4","c5","c6","c7","c8","c9","c10"]
        v_gates=["c5","c9"]
        l = 100
        s = 20
        res = 100
        meas = dot_score.dot_score_sample(pg.pygor,vols.tolist(), gates, v_gates, l, s, res)
        return [meas]

class ProblemSpec_edge(ProblemSpec):
    '''
    Random search on the specified edge
    '''
    def __init__(self, ndim, noise_var_obj, enable_feasibility=False, acq_criterion='EI', corner_th=200., edge_idxs=(), poff_idxs=()):

        self.ndim = ndim
        self.enable_feasibility = enable_feasibility
        self.acq_criterion = acq_criterion
        self.noise_var_obj = noise_var_obj
        self.corner_th = corner_th
        self.edge_idxs = list(edge_idxs)
        self.poff_idxs = list(poff_idxs)

        valid = all([poff_idx in edge_idxs for poff_idx in poff_idxs]) 
        if not valid:
            raise ValueError('poff_idx should be one of edge_idxs')

        ###
        # create GPs to predict the objective and the feasibility
        ###
        # for objective
        obj_max, obj_min = -1.0, 1.0
        l_prior_mean = 0.5 * np.ones(ndim)
        l_prior_var = 0.1*0.1 * np.ones(ndim)
        v_prior_mean = ((obj_max-obj_min)/4.0)**2
        v_prior_var = v_prior_mean**2

        self.gp_score = GP_util.create_GP(ndim, 'Matern52', v_prior_mean, l_prior_mean, (obj_max-obj_min)/2.0)
        GP_util.set_GP_prior(self.gp_score, l_prior_mean, l_prior_var, None, None) # do not set prior for kernel var
        GP_util.fix_hyperparams(self.gp_score, False, True) # fix kernel variance

        # for feasibility
        if not enable_feasibility:
            return
        l_prior_mean = 0.4 * np.ones(ndim)
        l_prior_var = 0.1*0.1 * np.ones(ndim)
        d_min, d_max = 0., 2000.
        v_prior_mean = ((d_max-d_min)/4.0)**2
        v_prior_var = v_prior_mean**2

        self.gp_dall = GP_util.create_GP_multiout(ndim, len(edge_idxs), 'Matern52', v_prior_mean, l_prior_mean, (d_max-d_min)/2.0, GP = GPyWrapper_MultiSeparate)

        GP_util.set_GP_prior(self.gp_dall, l_prior_mean, l_prior_var, None, None) # do not set prior for kernel var
        GP_util.fix_hyperparams(self.gp_dall, False, True) # fix kernel variance

    def set_data(self, u_all, data, do_opt=True):
        '''
        Set data for prediction, and optimize GPs
        Args:
            u_all: unit vector
            data: dictionary corresponding u_all
        Returns:
            Extra measurement used to evaluate the objective and feasibility
        '''
        self.u_all = u_all
        self.obj_data, meas = self.objective(u_all, data) # data for computing objective value. For this model, obj_data == obj_val
        self.obj_val = self.obj_data # objective value should be a scalar value
        self.gp_score.create_model(u_all, self.obj_data, self.noise_var_obj, noise_prior='fixed')

        if self.enable_feasibility:
            self.fea_data, meas_fea = self.feasibility(u_all, data)
            self.gp_dall.create_model(u_all, self.fea_data, np.square(10./2.))
            meas += meas_fea
        if do_opt:
            if self.enable_feasibility:
                self.gp_dall.optimize(num_restarts=20, opt_messages=False, print_result=True)
        return meas

    def objective(self, u_all, data):
        '''
        Evaluate the objective fuction
        Args:
            u_all: unit vector
            data: dictionary corresponding u_all
        Returns:
            real number, scalar (u_all.ndim==1) or 1D (u_all.ndim==2)
            empty list, meaning no extra measurement required
        '''

        if u_all.ndim == 2:
            score = np.zeros(u_all.shape[0])
        else:
            score = 0.0

        return score, []

    def feasibility(self, u_all, data):
        '''
        Evaluate the feasibility fuction

        Args:
            u_all: 1D (ndim) or 2D (num_vectors x ndim) array
            data: dictionary corresponding u_all
        Returns:
            selected dall
            empty list, meaning no extra measurement required
        '''
        fea_data = data['d_all'][:,self.edge_idxs]

        # add penalty for non-pinchoff
        if len(self.poff_idxs) != 0:
            found = data['found']
            poff_all = data['poff_all']
            for poff_idx in self.poff_idxs:
                good = np.logical_and(poff_all[:,poff_idx], found)
                gp_idx = self.edge_idxs.index(poff_idx)
                fea_data[:,gp_idx] += self.corner_th * np.logical_not(good)
        return fea_data, []

    def pred_feasibility(self, uvec):
        uvec = uvec[np.newaxis,:]
        uvec = uvec / np.sqrt(np.sum(np.square(uvec)))
        mean, var = self.gp_dall.predict_f(uvec, full_cov=False)
        std = np.sqrt(var)
        prob = np.prod(norm.cdf((self.corner_th-mean)/std), axis=-1)
        return prob[0]

    def acquisition_func(self, noise_var_next=None):
        '''
        Returns:
            acqusition function
        '''
        grad=False
        if noise_var_next is None:
            noise_var = self.noise_var_obj
        else:
            noise_var = noise_var_next

        if self.acq_criterion == 'EI':
            current_best_max = self.obj_data.max()
            func_EI = lambda x : EI_from_GP(self.gp_score, x, current_best_max, transform_x=Normalize(), min_problem=False, grad=grad) # negative sign for max -> min
            #func_EI = partial(EI_from_GP, gp=self.gp_score, best_prev=current_best_max, transform_x=make_length_one, min_problem=False, negative=True)
        elif self.acq_criterion == 'augmented_EI':
            post_mean, var_post = self.gp_d.predict_f(u_all, full_cov=False)
            idx_max = np.argmax(post_mean)
            max_post_mean = post_mean[idx_max,0] # array shape of post_mean is num_points x 1
            func_EI = lambda x : EI_from_GP(self.gp_d, x, max_post_mean, std_noise=np.sqrt(noise_var), transform_x=make_length_one, min_problem=False) # negative sign for max -> min
        else:
            raise ValueError('Incorrect acqusition criterion.')

        if self.enable_feasibility:
            #func_acq = lambda x: func_EI(x) * self.pred_feasibility(x)
            func_fea = lambda x : feasibility_from_GPR(self.gp_dall, x, self.corner_th, transform_x=Normalize(),grad=grad)
            func_acq = Negative_func(Prod_func([func_EI, func_fea]))
        else:
            func_acq = Negative_func(func_EI)
        return func_acq

    def do_extra_meas(self, pg, vols):
        gates = ["c3","c4","c5","c6","c7","c8","c9","c10"]
        v_gates=["c5","c9"]
        #gates = ["c3","c4","c8","c10","c11","c12","c16"]
        #v_gates=["c8","c12"]

        pygor = pg.pygor
        pygor.setvals(gates,vols.tolist())

        #score, measurements = score_driver.decision_aqu(pygor, vols, decision_function=0.1)
        score, measurements = score_driver.decision_aqu(pygor, vols, decision_function=0.05, low_res=16, high_res=48)
        print('Score: ', score)

        return [score] + measurements


class Prod_func(object):
    def __init__(self, func_list):
        self.func_list = func_list
    def __call__(self, x):
        result_all = [func(x) for func in self.func_list]
        isscalar_all = np.array([np.isscalar(result) for result in result_all])
        if np.all(isscalar_all):
            val = np.prod(result_all)
            return val
        elif np.all(np.logical_not(isscalar_all)):
            vals = np.array([result[0] for result in result_all])
            grads = np.array([result[1] for result in result_all])
            val = np.prod(vals) # product of  all values

            # gradient of f = f_1 * f_2 * f_3 * ...
            prod_except_i = prod_except_itself(vals)
            grad = np.sum(grads* prod_except_i[:,np.newaxis], axis=0)
            return val, grad
        else:
            raise ValueError('Some functions return grad, but others do not')

class Negative_func(object):
    def __init__(self, f):
        self.f = f
    def __call__(self, x):
        result = self.f(x)
        if np.isscalar(result):
            result = -1.0 * result
        else:
            result = tuple([-1.0*ele for ele in result])
        return result

def scan2d(pg, name_g1, name_g2, resol, vols, w1, w2, lb_g1, lb_g2, ub_g1, ub_g2):
    pg.set_params(vols.tolist())
    #idx_c5 = active_gate_names.index("c5")
    #idx_c9 = active_gate_names.index("c9")

    y_center = pg.getval(name_g1)
    x_center = pg.getval(name_g2)
    w_from = np.array([y_center-w1, x_center-w2])
    w_to = np.array([y_center+w1, x_center+w2])
    lb_w = np.array([lb_g1, lb_g2])
    ub_w = np.array([ub_g1, ub_g2])
    w_from, w_to = translate_window_inside_boundary(w_from, w_to, lb_w, ub_w)
    t = time.time()
    img=pg.do2d(name_g1,float(w_from[0]),float(w_to[0]),int(resol),
                name_g2,float(w_from[1]),float(w_to[1]),int(resol)).data
    return img

def do_extra_meas(pg, vols, threshold):
    #gates = ["c3","c4","c5","c6","c7","c8","c9","c10"]
    #v_gates=["c5","c9"]
    #score_object = final_score_cls(-1.8e-10,4.4e-10,5e-11,-1.4781e-10,150)

    gates = ["c3","c4","c8","c10","c11","c12","c16"]
    v_gates=["c8","c12"]
    score_object = final_score_cls(-1.5667e-12,1.172867e-10,9.5849571e-12,4.604e-12,150)

    #pygor = pg.pygor
    #pygor.setvals(gates,vols.tolist())
    pg.set_params(vols.tolist())
    if isinstance(pg, PygorRewire):
        vols = np.array(pg.convert_to_raw(vols))


    #score, measurements = score_driver.decision_aqu(pygor, vols, decision_function=0.1)
    #score, measurements = score_driver.decision_aqu(pygor, vols, gates, v_gates, score_object, decision_function=0.05, low_res=20, high_res=60)
    score, measurements = score_driver.decision_aqu(pg.pygor, vols, gates, v_gates, score_object, decision_function=threshold, low_res=16, high_res=48)
    print('Score: ', score)

    return [score] + measurements

def is_inside(point, box):
    '''
    Args:
        box: (box_lb, box_ub), where each element is a vector
    '''
    return np.all(np.logical_and(point >= box[0], point <= box[1]))


class DummyExtMeas(object):
    def __init__(self, box_peaks, box_goodscore, goodscore=1.):
        '''
        Args:
            box: (box_lb, box_ub), where each element is a vector
        '''
        self.box_peaks = box_peaks
        self.box_goodscore = box_goodscore
        self.goodscore = goodscore

    def eval_single_point(self, point):
        peaks = is_inside(point, self.box_peaks)
        goodscore = is_inside(point, self.box_goodscore) * self.goodscore
        return peaks, goodscore

    def __call__(self, point):
        '''
        Args:
            point: 1D array
        '''
        if point.ndim == 1:
            peaks, goodscore = self.eval_single_point(point)
            if peaks:
                dummy_meas = [None] * 10
                dummy_meas[3] = int(peaks)
                dummy_meas[8] = goodscore
            else:
                dummy_meas = [None] * 4
                dummy_meas[3] = int(peaks)
            return dummy_meas
        else:
            raise ValueError('point should be 1D.')
