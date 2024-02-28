from dataclasses import dataclass


@dataclass
class Param:
    # lower bound
    lb: int
    # upper bound
    ub: int
    # user distribution
    user_dist: str
    # user distribution random seed
    user_seed: int
    # item random seed
    item_seed: int
    # lambda, regularization parameter
    lamb: float
    # number of users per pair
    n_p: int
    # number of items
    m: int
    # dimension
    d: int
    # noise
    noise: str    # delta for the bound
    delta: float
    # epsilon for epsilon constraint
    epsilon: float
    # hyperplane seed
    h_seed: int
    # number of hyperplanes, -1 means using all hyperplanes
    n_h: int
    # opt method
    method: str
    # number of components
    K: int
    # number of samples per user
    T: int
    # number of EM iterations
    steps: int
    # EM learning rate (for phi)
    lr: float
    # epoch (for phi)
    epoch: int
    # random seed for the EM algorithm
    em_seed: int
    # name of the job
    job_name: str

    def __str__(self) -> str:
        s = ""
        for k, v in self.__dict__.items():
            s += str(v)
            s += "_"
        return s[:-1]