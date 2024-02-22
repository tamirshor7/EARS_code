
consts = {
    'frac_delay_length':(81, 40), # Length of the fractional delay filters used for RIR gen. fdl2 = (fdl-1) // 2
    'c': 343.0,
    'room_isinside_max_iter': 20, # Max iterations for checking if point is inside room 
    'eps': 1e-10,
    'eps_diagonals': 1e-6,
}

def inch_to_meter(inch_val):
    return inch_val * 0.0254