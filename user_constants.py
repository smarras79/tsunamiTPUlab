# Update these lines based on your GCP specifications
BUCKET = 'tpu-tsunami-bucket'
PROJECT_ID = 'intense-vault-364117'




PUBLIC_COLAB = True
TPU_WORKER = ''

# Floating point tolerance for timesteps.
TIMESTEP_EPS = 1e-5
# Floating point tolerance used in Saint-Venant step function.
SAINT_VENANT_EPS = 1e-1
CUTOFF_MAX = 1e15
_G = 9.8
# Manning coefficient defaults. NB: The simulation can be more accurate if a
# river mask is provided which defines th river region of the DEM. In that case,
# `MANNING_COEFF_FLOODPLAIN` is used in the non-river region (i.e., the

# floodplain region). In the Conawy example, we take a simpler approach and use
# `np.ones()` for the river mask, so `MANNING_COEFF_FLOODPLAIN` is unused.
MANNING_COEFF_FLOODPLAIN = .02
MANNING_COEFF_RIVER = .05
MANNING_COEFF = .0
# The dynamic states are:
#   h: the absolute height
#   q_x: The water flow in the x direction.
#   q_y: The water flow in the y direction.
#   t: The current simulation time.
#   dt: The timestep size. Note that `dt` is held constant in this simulation.
_H = 'h'
_Q_X = 'q_x'
_U_PREV = 'u_prev'
_Q_Y = 'q_y'
_V_PREV = 'v_prev'
_T = 't'
_DT = 'dt'

INIT_STATE_KEYS = [_H, _Q_X, _Q_Y]
STATE_KEYS = INIT_STATE_KEYS + [_T, _DT]

# The static states are:
#   m: The Manning coefficient matrix.
#   e: The water bed elevation.
# We also specify the boundaries using {L,R,T,B}_BOUNDARIES.
_M = 'm'
_E = 'e'

_I_L_BOUNDARY = 'i_left_boundary'
_I_R_BOUNDARY = 'i_right_boundary'
_I_T_BOUNDARY = 'i_top_boundary'
_I_B_BOUNDARY = 'i_bottom_boundary'

_O_L_BOUNDARY = 'o_left_boundary'
_O_R_BOUNDARY = 'o_right_boundary'
_O_T_BOUNDARY = 'o_top_boundary'
_O_B_BOUNDARY = 'o_bottom_boundary'

_M_L_BOUNDARY = 'm_left_boundary'
_M_R_BOUNDARY = 'm_right_boundary'
_M_T_BOUNDARY = 'm_top_boundary'
_M_B_BOUNDARY = 'm_bottom_boundary'

L_BOUNDARIES = (_I_L_BOUNDARY, _M_L_BOUNDARY, _O_L_BOUNDARY)
R_BOUNDARIES = (_I_R_BOUNDARY, _M_R_BOUNDARY, _O_R_BOUNDARY)
T_BOUNDARIES = (_I_T_BOUNDARY, _M_T_BOUNDARY, _O_T_BOUNDARY)
B_BOUNDARIES = (_I_B_BOUNDARY, _M_B_BOUNDARY, _O_B_BOUNDARY)

ADDITIONAL_STATE_KEYS = [
    _M, _E, *L_BOUNDARIES, *R_BOUNDARIES, *T_BOUNDARIES, *B_BOUNDARIES
]
SER_EXTENSION = 'ser'