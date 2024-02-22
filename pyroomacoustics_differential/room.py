# this script combines all functions to produce recordings, based on pyRoomAcoustics.
# no classes are used: from given parameters directly to recordings.

import numpy as np
from scipy.spatial import ConvexHull
from .geometry import side, intersection_2D_segments , intersection_segment_polygon_surface
from .geometry_onp import area, ccw3p, side_onp, intersection_2D_segments_onp , intersection_segment_polygon_surface_onp
from .consts import consts


def wall_side_onp(wall, p):
    p = np.array(p)
    try:
        if (wall['dim'] != p.shape[0]):
            raise NameError('Wall.side input error : dimension of p and the wall must match.')
    except Exception as e:
        print(f"Failed to compare wall and p. Got wall {wall} p {p}")
        raise e

    return side_onp(p, wall['corners'][:,0], wall['normal'])

def wall_side(wall, p):
    p = np.array(p)
    if (wall['dim'] != p.shape[0]):
        raise NameError('Wall.side input error : dimension of p and the wall must match.')

    return side(p, wall['corners'][:,0], wall['normal'])


def wall_intersection_onp(wall, p1, p2):
    '''
    Returns the intersection point between the wall and a line segment.
    
    :arg p1: (np.array dim 2 or 3) first end point of the line segment
    :arg p2: (np.array dim 2 or 3) second end point of the line segment
    
    :returns: (np.array dim 2 or 3 or None) intersection point between the wall and the line segment
    '''
    
    p1 = np.array(p1)
    p2 = np.array(p2)

    if (wall['dim'] == 2):
        return intersection_2D_segments_onp(p1, p2, wall['corners'][:,0], wall['corners'][:,1]) 

    if (wall['dim'] == 3):
        return intersection_segment_polygon_surface_onp(p1, p2, wall['corners_2d'], wall['normal'], wall['plane_point'], wall['plane_basis'])


def wall_intersection(wall, p1, p2):
    '''
    Returns the intersection point between the wall and a line segment.
    
    :arg p1: (np.array dim 2 or 3) first end point of the line segment
    :arg p2: (np.array dim 2 or 3) second end point of the line segment
    
    :returns: (np.array dim 2 or 3 or None) intersection point between the wall and the line segment
    '''
    if (wall['dim'] == 2):
        return intersection_2D_segments(p1, p2, wall['corners'][:,0], wall['corners'][:,1]) 

    if (wall['dim'] == 3):
        return intersection_segment_polygon_surface(p1, p2, wall['corners_2d'], wall['normal'], wall['plane_point'], wall['plane_basis'])

    
def walls_intersects(wall, p1, p2):
    '''
    Tests if the given line segment intersects the wall.
    
    :arg p1: (ndarray size 2 or 3) first endpoint of the line segment
    :arg p2: (ndarray size 2 or 3) second endpoint of the line segment
    
    :returns: (tuple size 3)
        (bool) True if the line segment intersects the wall
        (bool) True if the intersection happens at a border of the wall
        (bool) True if the intersection happens at the extremity of the segment
    '''
    
    if (wall['dim'] == 2):
        intersection, borderOfSegment, borderOfWall = intersection_2D_segments(p1, p2, wall['corners'][:,0], wall['corners'][:,1])

    if (wall['dim'] == 3):
        intersection, borderOfSegment, borderOfWall = intersection_segment_polygon_surface(p1, p2, wall['corners_2d'], wall['normal'],
                                                                                                wall['plane_point'], wall['plane_basis'])

    if intersection is None:
        intersects = False
    else:
        intersects = True
        
    return intersects, borderOfWall, borderOfSegment
        

def walls_intersects_onp(wall, p1, p2):
    '''
    Tests if the given line segment intersects the wall.
    
    :arg p1: (ndarray size 2 or 3) first endpoint of the line segment
    :arg p2: (ndarray size 2 or 3) second endpoint of the line segment
    
    :returns: (tuple size 3)
        (bool) True if the line segment intersects the wall
        (bool) True if the intersection happens at a border of the wall
        (bool) True if the intersection happens at the extremity of the segment
    '''
    
    if (wall['dim'] == 2):
        intersection, borderOfSegment, borderOfWall = intersection_2D_segments_onp(p1, p2, wall['corners'][:,0], wall['corners'][:,1])

    if (wall['dim'] == 3):
        intersection, borderOfSegment, borderOfWall = intersection_segment_polygon_surface_onp(p1, p2, wall['corners_2d'], wall['normal'],
                                                                                                wall['plane_point'], wall['plane_basis'])

    if intersection is None:
        intersects = False
    else:
        intersects = True
        
    return intersects, borderOfWall, borderOfSegment


def convex_hull(walls):

    ''' 
    Finds the walls that are not in the convex hull
    '''

    all_corners = []
    for wall in walls[1:]:
        all_corners.append(wall['corners'].T)
    X = np.concatenate(all_corners, axis=0)
    #print(f"input to convexhull {X}")
    convex_hull = ConvexHull(X, incremental=True)
    #print(f"convex hull: points {convex_hull.points} simplices {convex_hull.simplices}")
    #breakpoint()
    # Now we need to check which walls are on the surface
    # of the hull
    in_convex_hull = [False] * len(walls)
    for i, wall in enumerate(walls):
        # We check if the center of the wall is co-linear or co-planar
        # with a face of the convex hull
        point = np.mean(wall['corners'], axis=1)

        for simplex in convex_hull.simplices:
            if point.shape[0] == 2:
                # check if co-linear
                p0 = convex_hull.points[simplex[0]]
                p1 = convex_hull.points[simplex[1]]
                if ccw3p(p0, p1, point) == 0:
                    # co-linear point add to hull
                    in_convex_hull[i] = True

            elif point.shape[0] == 3:
                # Check if co-planar
                p0 = convex_hull.points[simplex[0]]
                p1 = convex_hull.points[simplex[1]]
                p2 = convex_hull.points[simplex[2]]

                normal = np.cross(p1 - p0, p2 - p0)
                if np.abs(np.inner(normal, point - p0)) < consts['eps']:
                    # co-planar point found!
                    in_convex_hull[i] = True

    obstructing_walls = np.array([i for i in range(len(walls)) if not in_convex_hull[i]], dtype=np.int32)
    return obstructing_walls


def create_wall(corners, absorption, name=None):
    corners = np.array(corners, dtype=np.float32)

    # set first corner as origin of plane
    plane_point = np.array(corners[:,0])

    if (corners.shape == (2, 2)):
        normal = np.array([(corners[:, 1] - corners[:, 0])[1], (-1)*(corners[:, 1] - corners[:, 0])[0]]) # delta_y , -delta_x
        dim = 2
    elif (corners.shape[0] == 3 and corners[0].shape[0] > 2):
        # compute the normal assuming the vertices are aranged counter
        # clock wise when the normal defines "up"
        i_min = np.argmin(corners[0,:])
        i_prev = i_min - 1 if i_min > 0 else corners.shape[1] - 1
        i_next = i_min + 1 if i_min < corners.shape[1] - 1 else 0
        normal = np.cross(corners[:, i_next] - corners[:, i_min], 
                                corners[:, i_prev] - corners[:, i_min])
        dim = 3

        # Compute a basis for the plane and project the corners into that basis
        plane_basis = np.zeros((3,2), dtype=np.float32) # np.zeros((3,2), order='F', dtype=np.float32)
        localx = np.array(corners[:,1]-plane_point)
        #plane_basis = index_update(plane_basis, index[:,0] ,localx/np.linalg.norm(localx))
        plane_basis = plane_basis.at[:,0].set(localx/np.linalg.norm(localx))
        localy = np.array(np.cross(normal, localx))
        #plane_basis = index_update(plane_basis, index[:,1] ,localy/np.linalg.norm(localy))
        plane_basis = plane_basis.at[:,1].set(localy/np.linalg.norm(localy))
        # corners = np.concatenate((
        #     [ np.dot(corners.T - plane_point, plane_basis[:,0]) ], 
        #     [ np.dot(corners.T - plane_point, plane_basis[:,1]) ]
        #     ))
        # np.array(corners, order='F', dtype=np.float32)

        a = np.expand_dims(np.dot(corners.T - plane_point, plane_basis[:,0]), 0)
        b = np.expand_dims(np.dot(corners.T - plane_point, plane_basis[:,1]), 0)
        corners_2d = np.concatenate((a,b))
        
    else:
        raise NameError('corners must be an np.array dim 2x2 or 3xN, N>2')
    
    normal = normal/np.linalg.norm(normal)
    
    if (name is not None):
        name = name
    
    wall = {'corners': corners, 'normal': normal, 'dim': dim,
            'absorption': absorption, 'name': name}
    
    if (corners.shape[0] == 3 and corners[0].shape[0] > 2):
        wall['corners_2d'] = corners_2d
        wall['plane_point'] = plane_point
        wall['plane_basis'] = plane_basis
    
    return wall


# create the room
def from_corners(corners, 
            absorption=0.,
            fs=44100,
            t0=0.,
            max_order=1,
            sigma2_awgn=None,
            e_absorption=None,
            single_wall:bool = False):
    
    print('Building the room')
    
    corners = np.array(corners)
    
    if (corners.shape[0] != 2 or corners.shape[1] < 3):
        raise ValueError('Arg corners must be more than two 2D points.')
    
    # change the order of the corners if the signed area is negative
    if (area(corners) <= 0):
        corners = corners[:,::-1]

    if e_absorption != None:
        # compatibility. Translate absorption from energy coefficient to the former definition
        # used by pyRoomAcoustics V0.3.1 (the version we rely on)
        # 1 - a1 == sqrt(1 - a2) <=> a1 == 1 - sqrt(1 - a2)
        absorption = 1 - np.sqrt(1 - np.array(e_absorption, dtype='float64'))
    else:
        absorption = np.array(absorption, dtype='float64')
    
    if (absorption.ndim == 0):
        absorption = absorption * np.ones(corners.shape[1])
    elif (absorption.ndim >= 1 and corners.shape[1] != len(absorption)):
        raise ValueError('Arg absorption must be the same size as corners or must be a single value.')

    walls = []
    for i in range(corners.shape[1]):
        walls.append(create_wall(np.array([corners[:, i], corners[:, (i+1)%corners.shape[1]]]).T, absorption[i])) #, "wall_"+str(i)))
    #print(f"walls created: {walls}")
    if single_wall:
        for i in [0,1,2]:
            walls[i]['absorption'] = 1

    # from room constructor
    # Compute the filter delay if not provided
    if t0 < (consts['frac_delay_length'][0]-1)/float(fs)/2:
        t0 = (consts['frac_delay_length'][0]-1)/float(fs)/2
    
    normals = np.array([wall['normal'] for wall in walls]).T
    corners = np.array([wall['corners'][:, 0] for wall in walls]).T
    absorption = np.array([wall['absorption'] for wall in walls])
    dim = walls[0]['dim']

    # mapping between wall names and indices
    # wallsId = {}
    # for i in range(len(walls)):
    #     if walls[i]['name'] is not None:
    #         wallsId[walls[i]['name']] = i

    # check which walls are part of the convex hull
    # this function uses scipy.spatial, which we've decided to implement as-is.
    obstructing_walls = convex_hull(walls)

    return {'obstructing_walls': obstructing_walls, 
        'dim': dim, 't0':t0, 'fs':fs,
        'absorption': absorption, 'normals': normals,
        'corners': corners, 'walls': walls}
    # return {'obstructing_walls': obstructing_walls, 
    #         'dim': dim, 'wallsId': wallsId, 't0':t0, 'fs':fs,
    #         'absorption': absorption, 'normals': normals,
    #         'corners': corners, 'walls': walls}


# New function
# create the room
def one_wall_from_corners(corners, 
            absorption=0.,
            fs=44100,
            t0=0.,
            max_order=1,
            sigma2_awgn=None,
            e_absorption=None):
    '''It creates a room with one wall.'''
    
    print('Building the room')
    
    corners = np.array(corners)
    if (corners.shape[0] != 2 or corners.shape[1] > 2):
        raise ValueError('Arg corners must be two 2D points.')
    

    if e_absorption != None:
        # compatibility. Translate absorption from energy coefficient to the former definition
        # used by pyRoomAcoustics V0.3.1 (the version we rely on)
        # 1 - a1 == sqrt(1 - a2) <=> a1 == 1 - sqrt(1 - a2)
        absorption = 1 - np.sqrt(1 - np.array(e_absorption, dtype='float64'))
    else:
        absorption = np.array(absorption, dtype='float64')
    
    if (absorption.ndim == 0):
        absorption = absorption * np.ones(corners.shape[1])
    elif (absorption.ndim >= 1 and corners.shape[1] != len(absorption)):
        raise ValueError('Arg absorption must be the same size as corners or must be a single value.')

    walls = []
    raise NotImplementedError
    walls.append(create_wall(np.array([corners[:, i], corners[:, (i+1)%corners.shape[1]]]).T, absorption[i])) #, "wall_"+str(i)))
    
    # from room constructor
    # Compute the filter delay if not provided
    if t0 < (consts['frac_delay_length'][0]-1)/float(fs)/2:
        t0 = (consts['frac_delay_length'][0]-1)/float(fs)/2
    
    normals = np.array([wall['normal'] for wall in walls]).T
    corners = np.array([wall['corners'][:, 0] for wall in walls]).T
    absorption = np.array([wall['absorption'] for wall in walls])
    dim = walls[0]['dim']

    # mapping between wall names and indices
    # wallsId = {}
    # for i in range(len(walls)):
    #     if walls[i]['name'] is not None:
    #         wallsId[walls[i]['name']] = i

    # check which walls are part of the convex hull
    # this function uses scipy.spatial, which we've decided to implement as-is.
    obstructing_walls = convex_hull(walls)

    return {'obstructing_walls': obstructing_walls, 
        'dim': dim, 't0':t0, 'fs':fs,
        'absorption': absorption, 'normals': normals,
        'corners': corners, 'walls': walls}
    # return {'obstructing_walls': obstructing_walls, 
    #         'dim': dim, 'wallsId': wallsId, 't0':t0, 'fs':fs,
    #         'absorption': absorption, 'normals': normals,
    #         'corners': corners, 'walls': walls}


def extrude(
        room,
        height,
        v_vec=None,
        absorption=0.):
    '''
    Creates a 3D room by extruding a 2D polygon. 
    The polygon is typically the floor of the room and will have z-coordinate zero. The ceiling

    Parameters
    ----------
    height : float
        The extrusion height
    v_vec : array-like 1D length 3, optionnal
        A unit vector. An orientation for the extrusion direction. The
        ceiling will be placed as a translation of the floor with respect
        to this vector (The default is [0,0,1]).
    absorption : float or array-like
        Absorption coefficients for all the walls. If a scalar, then all the walls
        will have the same absorption. If an array is given, it should have as many elements
        as there will be walls, that is the number of vertices of the polygon plus two. The two
        last elements are for the floor and the ceiling, respectively. (default 1)
    '''

    if room['dim'] != 2:
        raise ValueError('Can only extrude a 2D room.')

    # default orientation vector is pointing up
    if v_vec is None:
        v_vec = np.array([0., 0., 1.])

    # check that the walls are ordered counterclock wise
    # that should be the case if created from from_corners function
    nw = len(room['walls'])
    floor_corners = np.zeros((2,nw))
    #floor_corners[:,0] = room['walls'][0]['corners'][:,0]
    #floor_corners = index_update(floor_corners, index[:,0], room['walls'][0]['corners'][:,0])
    floor_corners = floor_corners.at[:,0].set(room['walls'][0]['corners'][:,0])
    ordered = True

    for iw, wall in enumerate(room['walls'][1:]):
        if not np.allclose(room['walls'][iw]['corners'][:,1], wall['corners'][:,0]):
            ordered = False
        #floor_corners = index_update(floor_corners, index[:,iw+1] , wall['corners'][:,0])
        floor_corners = floor_corners.at[:,iw+1].set(wall['corners'][:,0])
    if not np.allclose(room['walls'][-1]['corners'][:,1], room['walls'][0]['corners'][:,0]):
        ordered = False

    if not ordered:
        raise ValueError("The wall list should be ordered counter-clockwise, which is the case \
            if the room is created with Room.from_corners")

    # make sure the floor_corners are ordered anti-clockwise (for now)
    if (area(floor_corners) <= 0):
        floor_corners = np.fliplr(floor_corners)

    walls = []
    for i in range(nw):
        corners = np.array([
            np.hstack([floor_corners[:,i], 0]),
            np.hstack([floor_corners[:,(i+1)%nw], 0]),
            np.hstack([floor_corners[:,(i+1)%nw], 0]) + height*v_vec,
            np.hstack([floor_corners[:,i], 0]) + height*v_vec
        ]).T
        walls.append(create_wall(corners, room['walls'][i]['absorption'], name=str(i)))

    absorption = np.array(absorption)
    if absorption.ndim == 0:
        absorption = absorption * np.ones(2)
    elif absorption.ndim == 1 and absorption.shape[0] != 2:
        raise ValueError("The size of the absorption array must be 2 for extrude, for the floor and ceiling")

    floor_corners = np.pad(floor_corners, ((0, 1),(0,0)), mode='constant')
    ceiling_corners = (floor_corners.T + height*v_vec).T

    # we need the floor corners to ordered clockwise (for the normal to point outward)
    floor_corners = np.fliplr(floor_corners)

    walls.append(create_wall(floor_corners, absorption[0], name='floor'))
    walls.append(create_wall(ceiling_corners, absorption[1], name='ceiling'))

    room['walls'] = walls
    room['dim'] = 3

    # re-collect all normals, corners, absoption
    room['normals'] = np.array([wall['normal'] for wall in room['walls']]).T
    room['corners'] = np.array([wall['corners'][:, 0] for wall in room['walls']]).T
    room['absorption'] = np.array([wall['absorption'] for wall in room['walls']])

    # recheck which walls are in the convex hull
    room['obstructing_walls'] = convex_hull(room['walls'])

    return room