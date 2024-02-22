from tkinter import dnd
import numpy as onp
from .room import wall_intersection_onp, wall_side_onp, walls_intersects_onp 
import itertools
from scipy.signal import fftconvolve
from concurrent.futures import ThreadPoolExecutor
from .consts import consts 
from .geometry_onp import ccw3p

# NEW
#from jaxlib.xla_extension import DeviceArray, Array

def is_obstructed(room, source, p, imageId = 0):
    '''
    Checks if there is a wall obstructing the line of sight going from a source to a point.
    
    :arg source: (SoundSource) the sound source (containing all its images)
    :arg p: (onp.array size 2 or 3) coordinates of the point where we check obstruction
    :arg imageId: (int) id of the image within the SoundSource object
    
    :returns: (bool)
        False (0) : not obstructed
        True (1) :  obstructed
    '''
    
    imageId = int(imageId)
    if (onp.isnan(source['walls'][imageId])):
        genWallId = -1
    else:
        genWallId = int(source['walls'][imageId])
    
    # Only 'non-convex' walls can be obstructing
    for wallId in room['obstructing_walls']:
    
        # The generating wall can't be obstructive
        if(wallId != genWallId):
        
            # Test if the line segment intersects the current wall
            # We ignore intersections at boundaries of the line of sight
            #intersects, borderOfWall, borderOfSegment = self.walls[wallId].intersects(source.images[:, imageId], p)
            intersectionPoint, borderOfSegment, _ = wall_intersection_onp(room['walls'][wallId],source['images'][:, imageId], p)

            if (intersectionPoint is not None and not borderOfSegment):
                
                # Only images with order > 0 have a generating wall. 
                # At this point, there is obstruction for source order 0.
                #if (source.orders[imageId] > 0):
                if (source['orders'][imageId] > 0):
                    #breakpoint()
                    # BEFORE:
                    # imageSide = wall_side_onp(source['walls'][genWallId], (source['images'][:, imageId]))
                    try:
                        # imageSide = wall_side_onp(room['walls'][source['walls'][genWallId]], (source['images'][:, imageId]))
                        imageSide = wall_side_onp(room['walls'][genWallId], (source['images'][:, imageId]))
                    except Exception as e:
                        print(f"Could not compute imageSide. Got room {room} source {source} genWallId {genWallId}")
                        print(f"genWallId {genWallId} was generated from the list: {list(range(len(source['images'].shape[0])-1, -1, -1))}")
                        raise e
                
                    # Test if the intersection point and the image are at
                    # opposite sides of the generating wall 
                    # We ignore the obstruction if it is inside the
                    # generating wall (it is what happens in a corner)

                    # BEFORE:
                    # intersectionPointSide = wall_side_onp(source['walls'][genWallId], intersectionPoint)
                    # intersectionPointSide = wall_side_onp(room['walls'][source['walls'][genWallId]], intersectionPoint)
                    intersectionPointSide = wall_side_onp(room['walls'][genWallId], intersectionPoint)
                    if (intersectionPointSide != imageSide and intersectionPointSide != 0):
                        return True
                else:
                    return True
            
    return False


def is_visible(room, source, p, imageId = 0):
    '''
    Returns true if the given sound source (with image source id) is visible from point p.
    
    :arg source: (SoundSource) the sound source (containing all its images)
    :arg p: (onp.array size 2 or 3) coordinates of the point where we check visibility
    :arg imageId: (int) id of the image within the SoundSource object
    
    :return: (bool)
        False (0) : not visible
        True (1) :  visible
    '''
    p = onp.array(p)
    imageId = int(imageId)
    
    # Check if there is an obstruction
    if(is_obstructed(room, source, p, imageId)):
        return False
    
    if (source['orders'][imageId] > 0):
    
        # Check if the line of sight intersects the generating wall
        genWallId = int(source['walls'][imageId])

        # compute the location of the reflection on the wall
        intersection = wall_intersection_onp(room['walls'][genWallId], p, onp.array(source['images'][:, imageId]))[0]
        
        # if onp.all(intersection == room['walls'][genWallId]['corners'][:,0]) and source['orders'][imageId] % 2:
        #     return False
        
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

    lower = onp.amin(onp.concatenate([w['corners'] for w in room_walls], axis=1), axis=1)
    upper = onp.amax(onp.concatenate([w['corners'] for w in room_walls], axis=1), axis=1)

    return onp.column_stack((lower, upper))


def is_inside(room_walls, room_dim, p, include_borders = True):
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
    p = onp.array(p)
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
    
    bbox_center = onp.mean(bbox, axis=1)
    bbox_max_dist = onp.linalg.norm(bbox[:,1] - bbox[:,0]) / 2
    # re-run until we get a non-ambiguous result
    it = 0
    while it < consts['room_isinside_max_iter']: 

        # Get random point outside the bounding box
        random_vec = onp.random.randn(room_dim)
        # import jax.random as random
        # key = random.PRNGKey(0)
        # random_vec = random.normal(key=key, shape=(room_dim,))
        
        random_vec /= onp.linalg.norm(random_vec)
        p0 = bbox_center + 2 * bbox_max_dist * random_vec

        ambiguous = False  # be optimistic
        is_on_border = False  # we have to know if the point is on the boundary
        count = 0  # wall intersection counter
        for i in range(len(room_walls)):
            res_wall_intersects = walls_intersects_onp(room_walls[i], p0, p)
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
    :arg p: (onp.array size 2 or 3) coordinates of the point where we check visibility
    
    :returns: (int array) list of results of visibility for each image
        -1 : unchecked (only during execution of the function)
        0 (False) : not visible
        1 (True) : visible
    '''
    visibilityCheck = onp.zeros_like(source['images'][0], dtype=onp.int32) - 1
    if is_inside(room['walls'], room['dim'], onp.array(p)):
        # Only check for points that are in the room!
        for imageId in range(len(visibilityCheck)-1, -1, -1):
            #visibilityCheck = index_update(visibilityCheck, imageId, float(is_visible(room, source, p, imageId)))
            #visibilityCheck = visibilityCheck.at[imageId].set(float(is_visible(room, source, p, imageId)))
            visibilityCheck[imageId] = float(is_visible(room, source, p, imageId))
    else:
        # If point is outside, nothing is visible
        for imageId in range(len(visibilityCheck)-1, -1, -1):
            #visibilityCheck = index_update(visibilityCheck, imageId, 0.0)
            #visibilityCheck = visibilityCheck.at[imageId].set(0.0)
            visibilityCheck[imageId] = 0.0
    
    return visibilityCheck


def first_order_images(room, source_position):
    # projected length onto normal
    ip = onp.sum(room['normals'] * (room['corners'] - source_position[:, onp.newaxis]), axis=0)

    # projected vector from source to wall
    d = ip * room['normals']

    # compute images points, positivity is to get only the reflections outside the room
    images = source_position[:, onp.newaxis] + 2 * d[:, ip > 0]

    # collect absorption factors of reflecting walls
    damping = (1 - room['absorption'])[ip > 0].squeeze()

    # collect the index of the wall corresponding to the new image
    wall_indices = onp.arange(len(room['walls']))[ip > 0]

    return images, damping, wall_indices

def mic_on_room_diagonal_onp(mic_pos, room_corners):
    # TODO: make it more general - for any diagonal in a room and not only for shoebox
    if ccw3p(room_corners.T[0], room_corners.T[2], mic_pos) or ccw3p(room_corners.T[1], room_corners.T[3], mic_pos):
        return True
    return False

def image_source_model_onp(room, sources, mics_array, max_order=1):
    visibility = []

    # FIX: check if any microphone is on any diagonal in the room. If yes - add consts['eps'] to it.
    for idx_mic, mic_pos in enumerate(mics_array['R'].T):
        if mic_on_room_diagonal_onp(mic_pos, room['corners']):
            # NEW: check if the array is a jax array
            # if isinstance(mics_array['R'][0,idx_mic], DeviceArray) or isinstance(mics_array['R'][0,idx_mic], Array):
            #     mics_array['R'] = mics_array['R'].at[0,idx_mic].add(consts['eps_diagonals'])
            # else:
            
            # BEFORE:
            mics_array['R'][0,idx_mic] += consts['eps_diagonals']

    for source in sources:

        # pure python
        if max_order > 0:

            # generate first order images
            i, d, w = first_order_images(room, source['pos'])
            images = [i]
            damping = [d]
            generators = [-onp.ones(i.shape[1])]
            wall_indices = [w]

            # generate all higher order images up to max_order
            o = 1
            while o < max_order:
                # generate all images of images of previous order
                img = onp.zeros((room['dim'], 0))
                dmp = onp.array([])
                gen = onp.array([])
                wal = onp.array([])
                for ind, si, sd in zip(range(images[o-1].shape[1]), images[o - 1].T, damping[o - 1]):
                    i, d, w = first_order_images(room, si)
                    img = onp.concatenate((img, i), axis=1)
                    dmp = onp.concatenate((dmp, d * sd))
                    gen = onp.concatenate((gen, ind*onp.ones(i.shape[1])))
                    wal = onp.concatenate((wal, w))

                # sort
                ordering = onp.lexsort(img)
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

            o_len = onp.array([x.shape[0] for x in generators])
            # correct the pointers for linear structure
            for o in onp.arange(2, len(generators)):
                generators[o] += onp.sum(o_len[0:o-1])

            # linearize the arrays
            images_lin = onp.concatenate(images, axis=1)
            damping_lin = onp.concatenate(damping)
            generators_lin = onp.concatenate(generators)
            walls_lin = onp.concatenate(wall_indices)
            
            # store the corresponding orders in another array
            ordlist = []
            for o in range(len(generators)):
                ordlist.append((o+1)*onp.ones(o_len[o]))
            orders_lin = onp.concatenate(ordlist)

            # add the direct source to the arrays
            #source['images'] = onp.concatenate((source['pos'].T, images_lin), axis=1) 
            source['images'] = onp.concatenate((source['pos'][:,onp.newaxis], images_lin), axis=1)

            """
            source.damping = onp.concatenate(([1], damping_lin))
            source.generators = onp.concatenate(([-1], generators_lin+1)).astype(onp.int)
            source.walls = onp.concatenate(([-1], walls_lin)).astype(onp.int)
            source.orders = onp.array(onp.concatenate(([0], orders_lin)), dtype=onp.int)
            """
            source['damping'] = onp.hstack(([1], damping_lin))
            source['generators'] = onp.hstack(([-1], generators_lin+1))
            source['walls'] = onp.hstack(([-1], walls_lin))
            source['orders'] = onp.array(onp.hstack(([0], orders_lin)))
        else:
            # when simulating free space, there is only the direct source
            source['images'] = source['pos'][:,onp.newaxis]
            source['damping'] = onp.ones(1)
            source['generators'] = -onp.ones(1)
            source['walls'] = -onp.ones(1)
            source['orders'] = onp.zeros(1)

        # Then we will check the visibilty of the sources
        # visibility is a list with first index for sources, and second for mics

        # visibility.append(onp.ones((mics_array['R'].shape[1], source['images'].shape[1])))
        
        visibility.append([])

        #In general, we need to check for not shoebox rooms
        for mic in mics_array['R'].T:
            visibility[-1].append( check_visibility_for_all_images(room, source, mic) )
        
        visibility[-1] = onp.array(visibility[-1])

        I = onp.zeros(visibility[-1].shape[1], dtype=bool)
        for mic_vis in visibility[-1]:
            I = onp.logical_or(I, mic_vis == 1)

        # Now we can get rid of the superfluous images
        source['images'] = source['images'][:,I]
        source['damping'] = source['damping'][I]
        source['generators'] = source['generators'][I]
        source['walls'] = source['walls'][I]
        source['orders'] = source['orders'][I]
        
        visibility[-1] = visibility[-1][:,I]
    
    #print(f"Exiting from the function image_source_model_onp with sources {sources}")

        
    return visibility, sources


def fractional_delay(t0):
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
    return onp.hanning(fdl) * onp.sinc(onp.arange(fdl) - (fdl-1)/2 - t0)

def get_rir_onp(source, mic, visibility, fs, t0, max_order, t_max=None):
    '''
    # t0=0.
    Compute the room impulse response between the source
    and the microphone whose position is given as an
    argument.
    '''
    
    # fractional delay length (81,40)
    fdl, fdl2 = consts['frac_delay_length']
    #fdl2 = (fdl-1) // 2

    # compute the distance
    #dist = onp.sqrt(onp.sum((source['images'] - mic[:, onp.newaxis])**2, axis=0)) #instead of the func 'distance'
    dist = onp.linalg.norm(source['images'] - mic[:, onp.newaxis], axis=0)
    #np.sqrt(np.sum((self.images - ref_point[:, np.newaxis])**2, axis=0))
    # t0=0.013316751298034278
    time = dist / consts['c'] + t0
    alpha = source['damping'] / (4.*onp.pi*dist)

    # the number of samples needed
    if t_max is None:
        # we give a little bit of time to the sinc to decay anyway
        try:
            N = onp.ceil((1.05 * time.max() - t0) * fs)
        except:
            print(f"time: {time} source {source} mic {mic}")
            raise ValueError()
    else:
        N = onp.ceil((t_max - t0) * fs)

    N += fdl

    t = onp.arange(N) / float(fs)
    
    ir = onp.zeros(t.shape)

    ir_by_orders = {}
    for o in range(max_order+1):
        ir_by_orders[o] = onp.zeros_like(ir)

    # ir = onp.zeros(t.shape)
    # ir_source = onp.zeros_like(ir)
    # ir_images = onp.zeros_like(ir)

    for i in range(len(visibility)):
        if visibility[i] == 1:
            time_ip = int(onp.round(fs * time[i]))
            time_fp = (fs * time[i]) - time_ip
            #ir[time_ip-fdl2:time_ip+fdl2+1] += alpha[i]*self.fractional_delay(time_fp)
            #ir = index_add(ir, index[time_ip-fdl2:time_ip+fdl2+1], alpha[i] * fractional_delay(time_fp))
            # ir[time_ip-fdl2:time_ip+fdl2+1] += alpha[i] * fractional_delay(time_fp)

            addition_ir = alpha[i] * fractional_delay(time_fp)
            
            ir[time_ip-fdl2:time_ip+fdl2+1] += addition_ir
            
            ir_by_orders[int(source['orders'][i])][time_ip-fdl2:time_ip+fdl2+1] += addition_ir

    return ir, ir_by_orders


def compute_rir_threaded(sources, mics_array, visibility, room_fs, room_t0, max_order):
    rir = []
    rir_by_orders = {o:[] for o in range(max_order+1)}

    args_list = [(source, visibility[s], room_fs, room_t0, max_order) for s, source in enumerate(sources)]
    # here mic will contain the x and y coordinate of each microphone
    for m, mic in enumerate(mics_array['R'].T):
        h = []
        h_by_orders = {o:[] for o in range(max_order+1)}
        
        # for p in args_list:
        #     h_s = get_rir_onp(p[0], mic, p[1][m], p[2], p[3], p[4])
        #     h.append(h_s[0])
        #     for i in range(max_order+1):
        #         h_by_orders[i].append(h_s[1][i])

        ## PREVIOUSLY:
        # with ThreadPoolExecutor(max_workers=1) as pool:
        #     for h_s in pool.map(lambda p: get_rir_onp(p[0], mic, p[1][m], p[2], p[3], p[4]), args_list):
        #         h.append(h_s[0])
        #         for i in range(max_order+1):
        #             h_by_orders[i].append(h_s[1][i])

        # rir.append(h)
        # for i in range(max_order+1):
        #     rir_by_orders[i].append(h_by_orders[i])
        ## END PREVIOUSLY
        
        for p in args_list:
            h_s = get_rir_onp(p[0], mic, p[1][m], p[2], p[3], p[4])
            h.append(h_s[0])
            for i in range(max_order+1):
                h_by_orders[i].append(h_s[1][i])

        rir.append(h)
        for i in range(max_order+1):
            rir_by_orders[i].append(h_by_orders[i])

    return rir, rir_by_orders

def max_signal_len(sources, num_sources, fs):
    # calc max signal length
    f = lambda i: len(
        sources[i]['signal']) + onp.floor(sources[i]['delay'] * fs)
    max_sig_len = onp.array([f(i) for i in range(num_sources)]).max()
    # max_sig_len = int(fs * duration)

    return max_sig_len

def simulate_onp(rir_at_loc, sources, mics_array, room_fs):
    # based on pyRoomAcoustics.room.simulate()
    num_mics = mics_array['M']
    num_sources = len(sources)
    # compute the maximum signal length
    max_len_rir = onp.array([len(rir_at_loc[i][j])
                            for i, j in itertools.product(range(num_mics), range(num_sources))]).max()
    
    L = int(max_len_rir + max_signal_len(sources, num_sources, room_fs) - 1)
    if L % 2 == 1:
        L += 1

    # the array that will receive all the signals
    premix_signals = onp.zeros((num_sources, num_mics, L))

    # compute the signal at every microphone in the array
    for m in onp.arange(num_mics):
        for s in onp.arange(num_sources):

            sig = sources[s]['signal']
            if sig is None:
                continue

            d = int(onp.floor(sources[s]['delay'] * room_fs))
            h = rir_at_loc[m][s]
            premix_signals[s, m, d : d + len(sig) + len(h) - 1] += fftconvolve(h, sig)

    signals = onp.sum(premix_signals, axis=0)
    
    # super simplification of the return val
    return signals