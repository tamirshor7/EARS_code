import jax.numpy as jnp
import torch
from jax import jit, lax
from jax.scipy import signal
from EARS.pyroomacoustics_differential.room import wall_intersection, wall_side, walls_intersects
import itertools
from EARS.pyroomacoustics_differential.consts import consts
from EARS.pyroomacoustics_differential.geometry import ccw3p


def is_obstructed(room, source, p, imageId=0):
    '''
    Checks if there is a wall obstructing the line of sight going from a source to a point.

    :arg source: (SoundSource) the sound source (containing all its images)
    :arg p: (jnp.array size 2 or 3) coordinates of the point where we check obstruction
    :arg imageId: (int) id of the image within the SoundSource object

    :returns: (bool)
        False (0) : not obstructed
        True (1) :  obstructed
    '''

    imageId = int(imageId)
    if (jnp.isnan(source['walls'][imageId])):
        genWallId = -1
    else:
        genWallId = int(source['walls'][imageId])

    # Only 'non-convex' walls can be obstructing
    for wallId in room['obstructing_walls']:

        # The generating wall can't be obstructive
        if (wallId != genWallId):

            # Test if the line segment intersects the current wall
            # We ignore intersections at boundaries of the line of sight
            # intersects, borderOfWall, borderOfSegment = self.walls[wallId].intersects(source.images[:, imageId], p)
            intersectionPoint, borderOfSegment, borderOfWall = wall_intersection(room['walls'][wallId],
                                                                                 source['images'][:, imageId], p)

            if (intersectionPoint is not None and not borderOfSegment):

                # Only images with order > 0 have a generating wall.
                # At this point, there is obstruction for source order 0.
                if (source.orders[imageId] > 0):

                    imageSide = wall_side(source['walls'][genWallId], (source['images'][:, imageId]))

                    # Test if the intersection point and the image are at
                    # opposite sides of the generating wall
                    # We ignore the obstruction if it is inside the
                    # generating wall (it is what happens in a corner)
                    intersectionPointSide = wall_side(source['walls'][genWallId], intersectionPoint)
                    if (intersectionPointSide != imageSide and intersectionPointSide != 0):
                        return True
                else:
                    return True

    return False


def is_visible(room, source, p, imageId=0):
    '''
    Returns true if the given sound source (with image source id) is visible from point p.

    :arg source: (SoundSource) the sound source (containing all its images)
    :arg p: (jnp.array size 2 or 3) coordinates of the point where we check visibility
    :arg imageId: (int) id of the image within the SoundSource object

    :return: (bool)
        False (0) : not visible
        True (1) :  visible
    '''
    p = jnp.array(p)
    imageId = int(imageId)

    # Check if there is an obstruction
    if (is_obstructed(room, source, p, imageId)):
        return False

    if (source['orders'][imageId] > 0):

        # Check if the line of sight intersects the generating wall
        genWallId = int(source['walls'][imageId])

        # compute the location of the reflection on the wall
        intersection = wall_intersection(room['walls'][genWallId], p, jnp.array(source['images'][:, imageId]))[0]

        # the reflection point needs to be visible from the image source that generates the ray
        if intersection is not None:
            # Check visibility for the parent image by recursion
            return is_visible(room, source, intersection, source['generators'][imageId])
        else:
            return False
    else:
        return True


def get_bbox(room_walls):
    ''' Returns a bounding box for the room '''

    lower = jnp.amin(jnp.concatenate([w['corners'] for w in room_walls], axis=1), axis=1)
    upper = jnp.amax(jnp.concatenate([w['corners'] for w in room_walls], axis=1), axis=1)

    return jnp.column_stack((lower, upper))


def is_inside(room_walls, room_dim, p, include_borders=True):
    '''
    Checks if the given point is inside the room.

    Parameters
    ----------
    p: array_like, length 2 or 3
        point to be tested
    include_borders: bool, optional
        set true if a point on the wall must be considered inside the room

    Returns
    -------
        True if the given point is inside the room, False otherwise.
    '''

    p = jnp.array(p)
    if (room_dim != p.shape[0]):
        raise ValueError('Dimension of room and p must match.')

    # The method works as follows: we pick a reference point *outside* the room and
    # draw a line between the point to check and the reference.
    # If the point to check is inside the room, the line will intersect an odd
    # number of walls. If it is outside, an even number.
    # Unfortunately, there are a lot of corner cases when the line intersects
    # precisely on a corner of the room for example, or is aligned with a wall.

    # To avoid all these corner cases, we will do a randomized test.
    # We will pick a point at random outside the room so that the probability
    # a corner case happen is virtually zero. If the test raises a corner
    # case, we will repeat the test with a different reference point.

    # get the bounding box
    bbox = get_bbox(room_walls)
    bbox_center = jnp.mean(bbox, axis=1)
    bbox_max_dist = jnp.linalg.norm(bbox[:, 1] - bbox[:, 0]) / 2

    # re-run until we get a non-ambiguous result
    it = 0
    while it < consts['room_isinside_max_iter']:

        # Get random point outside the bounding box
        # random_vec = jnp.random.randn(self.dim)
        import jax.random as random
        key = random.PRNGKey(0)
        random_vec = random.normal(key=key, shape=(room_dim,))

        random_vec /= jnp.linalg.norm(random_vec)
        p0 = bbox_center + 2 * bbox_max_dist * random_vec

        ambiguous = False  # be optimistic
        is_on_border = False  # we have to know if the point is on the boundary
        count = 0  # wall intersection counter
        for i in range(len(room_walls)):
            res_wall_intersects = walls_intersects(room_walls[i], p0, p)
            intersects, border_of_wall, border_of_segment = res_wall_intersects
            # this flag is True when p is on the wall
            if border_of_segment:
                is_on_border = True
            elif border_of_wall:
                # the intersection is on a corner of the room
                # but the point to check itself is *not* on the wall
                # then things get tricky
                ambiguous = True

            # count the wall intersections
            if intersects:
                count += 1

        # start over when ambiguous
        if ambiguous:
            it += 1
            continue

        else:
            if is_on_border and not include_borders:
                return False
            elif is_on_border and include_borders:
                return True
            elif count % 2 == 1:
                return True
            else:
                return False

    # We should never reach this
    raise ValueError(
        ''' 
        Error could not determine if point is in or out in maximum number of iterations.
        This is most likely a bug, please report it.
        '''
    )


def check_visibility_for_all_images(room, source, p):
    '''
    Checks visibility from a given point for all images of the given source.

    This function tests visibility for all images of the source and returns the results
    in an array.

    :arg source: (SoundSource) the sound source object (containing all its images)
    :arg p: (jnp.array size 2 or 3) coordinates of the point where we check visibility

    :returns: (int array) list of results of visibility for each image
        -1 : unchecked (only during execution of the function)
        0 (False) : not visible
        1 (True) : visible
    '''

    visibilityCheck = jnp.zeros_like(source['images'][0], dtype=jnp.int32) - 1

    if is_inside(room['walls'], room['dim'], jnp.array(p)):
        # Only check for points that are in the room!
        for imageId in range(len(visibilityCheck) - 1, -1, -1):
            # visibilityCheck = index_update(visibilityCheck, imageId, float(is_visible(room, source, p, imageId)))
            visibilityCheck = visibilityCheck.at[imageId].set(float(is_visible(room, source, p, imageId)))
    else:
        # If point is outside, nothing is visible
        for imageId in range(len(visibilityCheck) - 1, -1, -1):
            # visibilityCheck = index_update(visibilityCheck, imageId, 0.0)
            visibilityCheck = visibilityCheck.at[imageId].set(0.0)

    return visibilityCheck


def first_order_images(room, source_position):
    # projected length onto normal
    ip = jnp.sum(room['normals'] * (room['corners'] - source_position[:, jnp.newaxis]), axis=0)

    # projected vector from source to wall
    d = ip * room['normals']

    # compute images points, positivity is to get only the reflections outside the room
    images = source_position[:, jnp.newaxis] + 2 * d[:, ip > 0]

    # collect absorption factors of reflecting walls
    damping = (1 - room['absorption'])[jnp.where(ip > 0)]

    # collect the index of the wall corresponding to the new image
    wall_indices = jnp.arange(len(room['walls']))[jnp.where(ip > 0)]

    return images, damping, wall_indices


def mic_on_room_diagonal(mic_pos, room_corners):
    # TODO: make it more general - for any diagonal in a room and not only for shoebox
    if ccw3p(room_corners.T[0], room_corners.T[2], mic_pos) or ccw3p(room_corners.T[1], room_corners.T[3], mic_pos):
        return True
    return False


def image_source_model(room, sources, mics_array, max_order=1):
    visibility = []

    # FIX: check if any microphone is on any diagonal in the room. If yes - add consts['eps'] to it.
    for idx_mic, mic_pos in enumerate(mics_array['R'].T):
        if mic_on_room_diagonal(mic_pos, room['corners']):
            # mics_array['R'][0,idx_mic] += consts['eps_diagonals']
            mics_array = mics_array['R'].at[0, idx_mic].add(consts['eps_diagonals'])

    for source in sources:
        # pure python
        if max_order > 0:

            # generate first order images
            i, d, w = first_order_images(room, source['pos'])
            images = [i]
            damping = [d]
            generators = [-jnp.ones(i.shape[1])]
            wall_indices = [w]

            # generate all higher order images up to max_order
            o = 1
            while o < max_order:
                # generate all images of images of previous order
                img = jnp.zeros((room['dim'], 0))
                dmp = jnp.array([])
                gen = jnp.array([])
                wal = jnp.array([])
                for ind, si, sd in zip(range(images[o - 1].shape[1]), images[o - 1].T, damping[o - 1]):
                    i, d, w = first_order_images(room, si)
                    img = jnp.concatenate((img, i), axis=1)
                    dmp = jnp.concatenate((dmp, d * sd))
                    gen = jnp.concatenate((gen, ind * jnp.ones(i.shape[1])))
                    wal = jnp.concatenate((wal, w))

                # sort
                ordering = jnp.lexsort(img)
                img = img[:, ordering]
                dmp = dmp[ordering]
                gen = gen[ordering]
                wal = wal[ordering]

                # add to array of images
                images.append(img)
                damping.append(dmp)
                generators.append(gen)
                wall_indices.append(wal)

                # next order
                o += 1

            o_len = jnp.array([x.shape[0] for x in generators])
            # correct the pointers for linear structure
            for o in jnp.arange(2, len(generators)):
                generators[o] += jnp.sum(o_len[0:o - 1])

            # linearize the arrays
            images_lin = jnp.concatenate(images, axis=1)
            damping_lin = jnp.concatenate(damping)
            generators_lin = jnp.concatenate(generators)
            walls_lin = jnp.concatenate(wall_indices)

            # store the corresponding orders in another array
            ordlist = []
            for o in range(len(generators)):
                ordlist.append((o + 1) * jnp.ones(o_len[o]))
            orders_lin = jnp.concatenate(ordlist)

            # add the direct source to the arrays
            # source['images'] = jnp.concatenate((source['pos'].T, images_lin), axis=1)
            source['images'] = jnp.concatenate((source['pos'][:, jnp.newaxis], images_lin), axis=1)

            """
            source.damping = jnp.concatenate(([1], damping_lin))
            source.generators = jnp.concatenate(([-1], generators_lin+1)).astype(jnp.int)
            source.walls = jnp.concatenate(([-1], walls_lin)).astype(jnp.int)
            source.orders = jnp.array(jnp.concatenate(([0], orders_lin)), dtype=jnp.int)
            """
            source['damping'] = jnp.hstack(([1], damping_lin))
            source['generators'] = jnp.hstack(([-1], generators_lin + 1))
            source['walls'] = jnp.hstack(([-1], walls_lin))
            source['orders'] = jnp.array(jnp.hstack(([0], orders_lin)))
        else:
            # when simulating free space, there is only the direct source
            source['images'] = source['pos'][:, jnp.newaxis]
            source['damping'] = jnp.ones(1)
            source['generators'] = -jnp.ones(1)
            source['walls'] = -jnp.ones(1)
            source['orders'] = jnp.zeros(1)

        # Then we will check the visibilty of the sources
        # visibility is a list with first index for sources, and second for mics

        # In general, we need to check for not shoebox rooms
        visibility.append([])

        for mic in mics_array['R'].T:
            visibility[-1].append(check_visibility_for_all_images(room, source, mic))
        visibility[-1] = jnp.array(visibility[-1])

        I = jnp.zeros(visibility[-1].shape[1], dtype=bool)
        for mic_vis in visibility[-1]:
            I = jnp.logical_or(I, mic_vis == 1)

        # Now we can get rid of the superfluous images
        source['images'] = source['images'][:, I]
        source['damping'] = source['damping'][I]
        source['generators'] = source['generators'][I]
        source['walls'] = source['walls'][I]
        source['orders'] = source['orders'][I]

        visibility[-1] = visibility[-1][:, I]

    return visibility, sources


@jit
def fractional_delay_jax(t0):
    '''
    Creates a fractional delay filter using a windowed sinc function.
    The length of the filter is fixed by the module wide constant
    `frac_delay_length` (default 81).

    Parameters
    ----------
    t0: float
        The delay in fraction of sample. Typically between -1 and 1.

    Returns
    -------
    numpy array
        A fractional delay filter with specified delay.
    '''
    fdl, _ = consts['frac_delay_length']
    # return jnp.hanning(N) * jnp.sinc(jnp.arange(N) - (N-1)/2 - t0)
    return jnp.hanning(fdl) * jnp.sinc(jnp.arange(fdl).tile((t0.shape[0], 1)) - (fdl - 1) / 2 - t0[:, jnp.newaxis])


@jit
def compute_attenuation_params(loc1, loc2, damping, t0):
    dist = jnp.linalg.norm(loc1 - loc2, axis=0)
    time = dist / consts['c'] + t0
    alpha = damping / (4. * jnp.pi * dist)
    return time, alpha


def get_rir_jax(source, mic, visibility, fs, t0=0., t_max=None):
    '''
    Compute the room impulse response between the source
    and the microphone whose position is given as an
    argument.
    '''

    # fractional delay length
    fdl, fdl2 = consts['frac_delay_length']
    # fdl2 = (fdl-1) // 2

    # compute the distance
    time, alpha = compute_attenuation_params(source['images'], mic[:, jnp.newaxis], source['damping'], t0)

    # the number of samples needed
    if t_max is None:
        # we give a little bit of time to the sinc to decay anyway
        N = jnp.ceil((1.05 * time.max() - t0) * fs)
    else:
        N = jnp.ceil((t_max - t0) * fs)

    N += fdl

    ir = jnp.zeros(int(N))

    time_ips = jnp.round(fs * time).astype(jnp.int64)
    time_fps = (fs * time) - time_ips
    frac_delay_all = alpha[:, jnp.newaxis] * fractional_delay_jax(time_fps)
    visibility_true_locations = jnp.where(visibility == 1)[0]
    for frac_delay, time_ip in zip(frac_delay_all[visibility_true_locations], time_ips[visibility_true_locations]):
        ir = ir.at[time_ip - fdl2:time_ip + fdl2 + 1].add(frac_delay)

    return ir


def compute_rir_jax(sources, mics_array, visibility, room_fs, room_t0):
    max_len_rir = 0
    hs = []
    for m, mic in enumerate(mics_array['R'].T):

        for s, source in enumerate(sources):
            cur_rir = get_rir_jax(source, mic, visibility[s][m], room_fs, room_t0)
            hs.append(cur_rir)
            max_len_rir = max(max_len_rir, len(cur_rir))

    # pad rir with zeros to make a numpy - will be used later on for convolving
    rir_arr = jnp.reshape(jnp.asarray([jnp.pad(i, (0, int(max_len_rir) - len(i))) for i in hs]),
                          (mics_array['M'], len(sources), max_len_rir))
    return rir_arr


def max_signal_len(sources, num_sources, fs):
    # calc max signal length
    f = lambda i: len(
        sources[i]['signal']) + jnp.floor(sources[i]['delay'] * fs)
    max_sig_len = jnp.array([f(i) for i in range(num_sources)]).max()
    # max_sig_len = int(fs * duration)

    return max_sig_len


def max_len_rir(rir_at_loc, num_mics, num_sources):
    return jnp.array([len(rir_at_loc[i][j])
                      for i, j in itertools.product(range(num_mics), range(num_sources))]).max()


# python forward_model_rir_single_time.py -gpu 3 -exp_name tst -opt_harmonies 0.5 1 2 3 -opt_harmonies_init_vals 0 0 0 0 -opt_phi_0_init_vals 0 0 0 0 -num_sources_power 4 -channel 0 -duration 1. -step_log_signals 1 -num_phases_mics 4 -optimize


@jit
def batch_convolve1d(x, y):
    # x: 1d array
    # y: batch of N 1d arrays
    # out: Nx(np.convolve(x,y[i]).shape[0])
    # based on convolve1d implemented as in https://github.com/google/jax/issues/1561
    x_reshaped = jnp.reshape(x,(x.shape[0], 1, x.shape[-1]))
    y_flipped = jnp.flip(jnp.reshape(y, (y.shape[0], 1, y.shape[1])), 2)

    # holy batch
    res = lax.conv_general_dilated(
        x_reshaped,
        y_flipped,
        window_strides=(1,),
        padding=[(y.shape[1] - 1, y.shape[1] - 1)])  # equivalent of padding='full' in NumPy

    return res


def mb_modified_simulate(rir_at_loc, signals, delay_sources, num_sources, num_mics, max_rec_len):
    # based on pyRoomAcoustics.room.simulate()

    # compute the maximum signal length
    # max_rec_len = max_len_rir + max_signal_len

    # the array that will receive all the signals
    premix_signals = jnp.zeros((max_rec_len.shape[0],num_sources, num_mics, max_rec_len.max()))

    for s in range(num_sources):
        sig = signals[:,s]

        h = rir_at_loc[:, s]

        conv_res = batch_convolve1d(sig, h)

        # premix_signals = index_add(premix_signals, index[s, :, delay_sources:delay_sources + len(sig) + h.shape[1] - 1], conv_res)
        premix_signals = premix_signals.at[:,s, :, delay_sources:delay_sources + sig.shape[-1] + h.shape[1] - 1].add(conv_res)


    signals = jnp.sum(premix_signals, axis=1)

    # super simplification of the return val
    return signals

def torch_mb_modified_simulate(rir_at_loc, signals, delay_sources, num_sources, num_mics, max_rec_len):
    # delay_sources is np array = 0
    # based on pyRoomAcoustics.room.simulate()

    # signals: (batch_size, num_sources, max_signal_len)

    # compute the maximum signal length
    # max_rec_len = max_len_rir + max_signal_len

    # the array that will receive all the signals
    # rir has shape (512,256,95) (num_mics, num_sources, max_len_rir)
    rir_at_loc = torch.tensor(rir_at_loc, device=signals.device)
    # why using max_rec_len.max()?
    # MEMO: simulated_recordings has shape (batch_size, num_mics, max_rec_len) is this what we want?
    # premix_signals: (batch_size, num_sources, num_mics, max_rec_len)
    premix_signals = torch.zeros((max_rec_len.shape[0],num_sources, num_mics, max_rec_len.max()), device=signals.device, dtype=torch.double)

    for s in range(num_sources):
        # sig has shape (batch_size, signal_len)
        sig = signals[:,s]

        # h has shape (num_mics, max_len_rir)
        h = rir_at_loc[:, s]
        # TO BE CHECKED: Is this the real equivalent of batch_conlvolve1d(sig,h)?
        # sig.unsqueeze(1) has shape (batch_size, 1, signal_len)
        # h.unsqueeze(1) has shape (num_mics, 1, rir_len) (rir_len = 95)
        #conv_res = torch.nn.functional.conv1d(sig.unsqueeze(1).to(torch.double), h.unsqueeze(1).to(torch.double),padding=h.shape[1] - 1)
        conv_res = torch_batch_convolve1d_multiple_microphones(h,sig) 
        # premix_signals = index_add(premix_signals, index[s, :, delay_sources:delay_sources + len(sig) + h.shape[1] - 1], conv_res)
        # delay_sources is 0
        # sig.shape[-1] (1373, depends on data) + h.shape[1] (95) - 1 = 95
        premix_signals[:,s, :, delay_sources:delay_sources + sig.shape[-1] + h.shape[1] - 1] += conv_res
        


    # PROBLEM: signals has shape (batch_size, num_mics, max_rec_len)
    # do we really need num_mics?
    signals = torch.sum(premix_signals, axis=1)

    # super simplification of the return val
    return signals
def torch_batch_convolve1d_multiple_microphones(x, y):
    '''This function computes CONVOLUTION (not cross-correlation as pytorch does) 
    of x (rir with multiple microphones) with each 1d array in y (signal)
    :param x: (num_mics, rir_len)
    :param y: (batch_size, signal_len)
    '''
    x = x.reshape(x.shape[0], 1, x.shape[1])
    # shape of x: (out_channels=num_mics, in_channels=1, kernel_size=rir_len) 
    x_flipped = torch.flip(x, [2])
    # shape of y: (batch_size, in_channels=1, signal_len)
    y = y.reshape(y.shape[0], 1, y.shape[1])
    #y_flipped = y.reshape(y.shape[0], 1, y.shape[1])
    # shape after conv: (batch_size, out_channels=num_mics, signal_len + kernel_size - 1)
    # we squeeze the result so that it will match the size of the array we want to add to
    res = torch.nn.functional.conv1d(y, x_flipped, padding=x.shape[-1]-1)
    return res
def torch_batch_convolve1d_single_microphone(x, y):
    '''This function computes CONVOLUTION (not cross-correlation as pytorch does) 
    of x (rir) with each 1d array in y (signal)
    :param x: (rir_len)
    :param y: (batch_size, signal_len)
    '''
    # shape of x: (out_channels, in_channels, kernel_size) 
    x_reshaped = torch.flip(x.reshape(1,1, x.shape[0]), [2])
    # shape of y: (batch_size, in_channels, signal_len)
    y_flipped = y.reshape(y.shape[0], 1, y.shape[1])
    # shape after conv: (batch_size, out_channels, signal_len + kernel_size - 1)
    # we squeeze the result so that it will match the size of the array we want to add to
    res = torch.nn.functional.conv1d(y_flipped, x_reshaped, padding=x.shape[0]-1).squeeze(1)
    return res
    

def torch_mb_modified_simulate_single_microphone(rir_at_loc, signals, delay_sources, num_sources, num_mics, max_rec_len):
    '''
    This function is the same as torch_mb_modified_simulate but it only returns the signal of the first microphone
    '''
    # delay_sources is np array = 0
    # based on pyRoomAcoustics.room.simulate()

    # signals: (batch_size, num_sources, max_signal_len)

    # compute the maximum signal length
    # max_rec_len = max_len_rir + max_signal_len

    # the array that will receive all the signals
    # before rir had shape (512,256,95) (num_mics, num_sources, max_len_rir)
    rir_at_loc = torch.tensor(rir_at_loc[0], device=signals.device)
    # now rir has shape (256,95) (num_sources, max_len_rir)

    # premix_signals: (batch_size, num_sources, max_rec_len)
    premix_signals = torch.zeros((max_rec_len.shape[0],num_sources, max_rec_len.max()), device=signals.device, dtype=torch.double)
    for s in range(num_sources):
        # sig has shape (batch_size, signal_len)
        sig = signals[:,s]

        # h has shape (max_len_rir)
        h = rir_at_loc[s]
        
        conv_res = torch_batch_convolve1d_single_microphone(h, sig)

        # delay_sources (0) + sig.shape[-1] (signal_len) + h.shape[0] (95) - 1 is the length of the convolved signal
        premix_signals[:,s,delay_sources:delay_sources + sig.shape[-1] + h.shape[0] - 1] += conv_res
        
    signals = torch.sum(premix_signals, axis=1)

    return signals

def torch_mb_modified_simulate_multi_microphones(rir_at_loc, signals, delay_sources, num_sources, num_mics, max_rec_len, use_all_distances=False):
    '''
    This function is the same as torch_mb_modified_simulate but it only returns the signal of the first microphone
    '''
    # delay_sources is np array = 0
    # based on pyRoomAcoustics.room.simulate()

    # signals: (batch_size, num_sources, max_signal_len)

    # compute the maximum signal length
    # max_rec_len = max_len_rir + max_signal_len

    # the array that will receive all the signals
    # rir has shape (8,256,95) (num_mics, num_sources, max_len_rir)
    rir_at_loc = torch.tensor(rir_at_loc, device=signals.device)
    

    # premix_signals: (batch_size, num_sources, num_mics, max_rec_len)
    premix_signals = torch.zeros((max_rec_len.shape[0],num_sources,num_mics, max_rec_len.max()), device=signals.device, dtype=torch.double)
    for s in range(num_sources):
        # Before slicing sig has shape (batch_size, num_sources, signal_len)
        # ATTENTION: control that the shape of the signal is correct!
        sig = signals[:,s]

        # h has shape (num_mics, max_len_rir)
        h = rir_at_loc[:,s]
        
        conv_res = torch_batch_convolve1d_multiple_microphones(h, sig)

        # delay_sources (0) + sig.shape[-1] (signal_len) + h.shape[0] (95) - 1 is the length of the convolved signal
        premix_signals[:,s,:,delay_sources:delay_sources + sig.shape[-1] + h.shape[1] - 1] += conv_res
        
    signals = torch.sum(premix_signals, axis=1)
    if not use_all_distances:
        return signals
    else:
        return signals, premix_signals

def simulate(rir_at_loc, sources, mics_array, room_fs):
    # based on pyRoomAcoustics.room.simulate()
    num_mics = mics_array['M']
    num_sources = len(sources)
    # compute the maximum signal length
    max_len_rir = jnp.array([len(rir_at_loc[i][j])
                             for i, j in itertools.product(range(num_mics), range(num_sources))]).max()

    L = int(max_len_rir + max_signal_len(sources, num_sources, room_fs) - 1)
    if L % 2 == 1:
        L += 1

    # the array that will receive all the signals
    premix_signals = jnp.zeros((num_sources, num_mics, L))

    # compute the signal at every microphone in the array
    for m in jnp.arange(num_mics):
        for s in jnp.arange(num_sources):

            sig = sources[s]['signal']
            if sig is None:
                continue

            d = int(jnp.floor(sources[s]['delay'] * room_fs))
            h = rir_at_loc[m][s]
            # premix_signals[s, m, d : d + len(sig) + len(h) - 1] += fftconvolve(h, sig)
            conv_res = signal.convolve(h,
                                       sig)  # prior the implementation of jax.scipy.signal.convolve was using jnp.convolve

            # premix_signals = index_add(premix_signals, index[s,m,d:d + len(sig) + len(h) - 1], conv_res)
            premix_signals = premix_signals.at[s, m, d:d + len(sig) + len(h) - 1].add(conv_res)

    # # pad with zeros
    # premix_signals_padded = jnp.zeros((num_sources, num_mics, max_recordings_len))
    # premix_signals_padded[:,:,:L] = premix_signals

    signals = jnp.sum(premix_signals, axis=0)

    # super simplification of the return val
    return signals