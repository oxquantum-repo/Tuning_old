import numpy as np

# n-shpere

def to_ndarray(obj):
    if not isinstance(obj, np.ndarray):
        obj = np.array(obj)
    return obj

def cart2sphere_1d(cart):
    cart_r = cart[::-1]
    cart_cumL2 = np.sqrt(np.cumsum(np.square(cart_r),axis=-1))
    r = cart_cumL2[-1]
    theta_r = np.arccos(cart_r[1:] / cart_cumL2[1:])
    if cart_r[0] < 0.0:
        theta_r[0] = 2.0*np.pi - theta_r[0]
    return theta_r[::-1], r
def cart2sphere_2d(cart):
    cart_r = cart[:,::-1]
    cart_cumL2 = np.sqrt(np.cumsum(np.square(cart_r),axis=-1))
    r = cart_cumL2[:,-1]
    theta_r = np.arccos(cart_r[:,1:] / cart_cumL2[:,1:])
    theta_r[cart_r[:,0]<0.0,0] = 2*np.pi - theta_r[cart_r[:,0]<0.0,0]
    return theta_r[:,::-1], r

def cart2sphere(cart):
    cart = to_ndarray(cart)
    if cart.ndim not in (1,2) or cart.shape[-1] < 2:
        raise ValueError('Input should be a vector or matrix (# of vectors X vector dimension(>=2) )')
    if cart.ndim == 1:
        return cart2sphere_1d(cart)
    else:
        return cart2sphere_2d(cart)

def sphere2cart_1d(theta, r):
    cart = np.stack((r,)*(theta.shape[-1]+1),axis=-1)
    cart[:-1] = cart[:-1] * np.cos(theta)
    cart[1:] = cart[1:] * np.cumprod(np.sin(theta),axis=-1)
    return cart
def sphere2cart_2d(theta, r):
    cart = np.stack((r,)*(theta.shape[-1]+1),axis=-1)
    cart[:,:-1] = cart[:,:-1] * np.cos(theta)
    cart[:,1:] = cart[:,1:] * np.cumprod(np.sin(theta),axis=-1)
    return cart

def sphere2cart(theta, r):
    theta = to_ndarray(theta)
    r = to_ndarray(r)
    if r.size == 1:
        if theta.ndim == 2:
            r = r.reshape((1,))
    if theta.ndim not in (1,2) or r.ndim != theta.ndim-1:
        raise ValueError('Theta should be a vector or matrix (r=1-less dim. than theta) ')
    if theta.ndim == 1:
        return sphere2cart_1d(theta, r)
    else:
        return sphere2cart_2d(theta, r)
    return cart

class PinchoffDetectorThreshold_(object):
    def __init__(self, threshold):
        self.threshold = threshold
    def __call__(self, trace, reverse_direction=False):
        low = trace < self.threshold
        if reverse_direction is True: #reverse search
            trace = trace[::-1]
        high_prev = np.concatenate( ([True], np.logical_not(low[:-1])), axis=0 ) # indicating whether it was above the thresholdshold right before
        change_points = np.logical_and(low,high_prev)
        keep_low = np.concatenate( (low[1:], [True]), axis=0 ) # indicating whether it keeps low afterwards
        for i in range (trace.size-2,-1,-1):
            keep_low[i] = np.logical_and(keep_low[i], keep_low[i+1])
        idxs = np.where(np.logical_and(change_points,keep_low))[0]
        if np.size(idxs) == 0:
            return -1

        idx = idxs[0]
        if reverse_direction is True: #reverse search
            idx = np.size(trace)-idx-1
        return idx

class PinchoffDetectorThreshold(object):
    def __init__(self, th_low):
        self.th_low = th_low
    def __call__(self, trace, reverse_direction=False):
        if reverse_direction is True: #reverse search
            trace = trace[::-1]

        low = trace < self.th_low
        # change_points is 1 when ~low -> low, or low -> ~low (~: not)
        #change_points = np.logical_xor(low[:-1], np.logical_not(low[1:]))
        change_points = np.logical_xor(low[:-1], low[1:])
        change_points = np.concatenate( ([True], change_points), axis=0 ) # change point is true for the first point

        possible_points = np.logical_and(low,change_points)

        # high_prev indicates whether it was above the thresholdshold at least once
        # (not high enough -> low is not pinchoff)
        #high_prev = np.concatenate( ([False], (trace>self.th_high)[:-1]), axis=0 ) 
        #for i in range(1,trace.size):
        #    high_prev[i] = np.logical_or(high_prev[i-1], high_prev[i])
        #possible_points = np.logical_and(possible_points,high_prev)

        # keep_low indicates whether it keeps low afterwards
        keep_low = np.concatenate( (low[1:], [True]), axis=0 ) 
        for i in range(trace.size-2,-1,-1):
            keep_low[i] = np.logical_and(keep_low[i], keep_low[i+1])

        idxs = np.where(np.logical_and(possible_points,keep_low))[0]
        if np.size(idxs) == 0:
            return -1

        idx = idxs[0]
        if reverse_direction is True: #reverse search
            idx = np.size(trace)-idx-1
        return idx

class ConductingDetectorThreshold(object):
    def __init__(self, th_high):
        self.th_high = th_high
    def __call__(self, trace):
        high = trace > self.th_high
        idxs = np.where(high)[0]
        if np.size(idxs) == 0:
            return -1
        return idxs[0] # return the first index satistying the condition

def merge_data(vols_poff_all, detected_all, vols_poff_axes_all, poff_all):
    vols_poff_all = np.array(vols_poff_all)
    detected_all = np.array(detected_all)
    vols_poff_axes_all = np.array(vols_poff_axes_all)
    poff_all = np.array(poff_all)

    if vols_poff_axes_all.size == 0 and poff_all.size == 0:
        return vols_poff_all, detected_all
    else:
        num_gates = vols_poff_all.shape[-1]
        vols_poff_axes_all = vols_poff_axes_all.reshape((-1, num_gates))
        poff_all = poff_all.ravel()

        vols_all = np.concatenate([vols_poff_all, vols_poff_axes_all], axis=0)
        found_all = np.concatenate([detected_all, poff_all], axis=0)
        return vols_all, found_all

def L1_norm(arr, axis=None, keepdims=False):
    return np.sum(np.fabs(arr), axis=axis, keepdims=keepdims)

def L2_norm(arr, axis=None, keepdims=False):
    return np.sqrt(np.sum(np.square(arr), axis=axis, keepdims=keepdims))

def ur_from_vols_origin(vols, origin, returntype='list'):
    vols = np.array(vols)
    num_data = vols.shape[0]
    if num_data == 0:
        return [], []
    ndim = vols.shape[1]

    if np.isscalar(origin):
        origin = origin * np.ones(ndim)

    diff = vols - origin[np.newaxis, :]
    r_all = L2_norm(diff, axis=1)
    u_all = diff / r_all[:,np.newaxis]

    if returntype == 'list':
        return [u for u in u_all], r_all.tolist()
    else:
        return u_all, r_all

def compute_hardbound(poff_vec, found, vols_pinchoff, step_back, axes, origin):
    do_change = False
    if len(axes) == 0:
        return do_change, origin

    ## No gate can pinchoff
    #if np.sum(poff_vec[axes]) == 0 and found:
    #    do_change = True
    #    new_origin = origin.copy()
    #    new_origin = vols_pinchoff + step_back
    #    return do_change, new_origin

    # Only one gate can pinchoff
    if np.sum(poff_vec[axes]) == 1:
        do_change = True
        new_origin = origin.copy()
        # if only one gate can pinchoff the current, move the origin
        idx = np.nonzero(poff_vec)
        if found:
            new_origin[idx] = vols_pinchoff[idx] + step_back
        if not found:
            new_origin[idx] = vols_pinchoff[idx]
        return do_change, new_origin

    return do_change, origin
