from EARS.localization.python_projection import Evaluator
from EARS.localization.python_projection import utils
class Constraints:
    '''Kinematic Constraints Class
    evaluator - evaluation instance
    grad_upperbound - velocity/accelaration upperbound
    dt - discretization step
    cfg - binary value stating first or second order constraint
    '''

    def __init__(self,evaluator:Evaluator, grad_upperbound, dt,cfg):
        self.eval = evaluator
        self.dt = dt
        self.bound = grad_upperbound
        self.Trans_operator = utils.secondDerivative if cfg else utils.primeT
        self.grad_operator = utils.secondDerivative if cfg else utils.firstDerivative



