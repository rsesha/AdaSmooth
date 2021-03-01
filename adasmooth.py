import numpy as np


class AdaSmooth:
    def __init__(self, 
                 model, 
                 get_params_flat, 
                 set_params_flat, 
                 stepsize=1e-4, 
                 beta1=0.9, 
                 beta2=0.999, 
                 epsilon=1e-12, 
                 alpha=0.95, 
                 momentum=0.9):
        self.model = model
        self.t = 0
        self.stepsize = stepsize
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.alpha = alpha
        self.m = np.zeros(self.dim, dtype=np.float32)
        self.v1 = np.zeros(self.dim, dtype=np.float32)
        self.v2 = np.zeros(self.dim, dtype=np.float32)
        
        self.get_params_flat = get_params_flat
        self.set_params_flat = set_params_flat
        self.dim = len(self.get_params_flat(model))

    def update(self, grad):
        self.t += 1
        step = self._compute_step(grad)
        theta = self.get_params_flat(self.model)
        self.set_params_flat(self.model, theta + step)

    def _compute_step(self, grad):
        # calculate first moment of gradient (momemtum)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad

        # calculate second moment of gradient (RMSprop)
        # use 'np.square(grad - m_t)' for Adabelief instead of 'np.square(grad)'
        self.v1 = self.beta2 * self.v1 + (1 - self.beta2) * np.square(grad - self.m)

        # correct bias (mostly affects initial steps)
        m_corr_t = self.m / (1.0 - np.pow(self.beta1, self.t))
        v_corr_t = self.v1 / (1.0 - np.pow(self.beta2, self.t))

        # calculate adaptive step
        adaptive_step = m_corr_t / (np.sqrt(v_corr_t) + self.epsilon)
        
        # calculate SGD step
        self.v2 = self.momentum * self.v2 + (1. - self.momentum) * grad
        sgd_step = -self.stepsize * self.v2
        
        # calculated weighted average step
        split_factor = np.power(self.alpha, self.t)
        step = split_factor * sgd_step + adaptive_step * (1 - split_factor)

        # apply lr
        return self.stepsize * step
