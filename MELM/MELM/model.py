class model:
    def __init__(self, L,w,b,beta,mu,sigma,adaw,n_types):
        self.L = L
        self.w = w
        self.b = b
        self.beta=beta
        self.mu=mu
        self.sigma=sigma
        self.ada_weights = adaw
        self.n_types=n_types

