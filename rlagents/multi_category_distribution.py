from torch.distributions import Distribution, Categorical, constraints
from torch.distributions.utils import lazy_property
import torch 


class MultiCategoryDistribution(Distribution):
    arg_constraints = {'probs_1': constraints.simplex,
                       'logits_1': constraints.real_vector,
                       'probs_2': constraints.simplex, 
                       'logits_2': constraints.real_vector}
 
    def __init__(self, 
                 probs_12 = None,
                #  probs_1=None, 
                #  probs_2=None, 
                 logits_1=None, 
                 logits_2=None,
                 validate_args=None):
        probs_1 = probs_12[:, :9]
        probs_2 = probs_12[:, 9:]
                  
        self.distribution_1 = Categorical(probs_1, logits_1)
        self.distribution_2 = Categorical(probs_2, logits_2)
        
        self._param_1 = self.probs_1 if probs_1 is not None else self.logits_1
        self._param_2 = self.probs_2 if probs_2 is not None else self.logits_2 
        self._num_events = (self._param_1.size()[-1]) * (self._param_2.size()[-1])
        
        batch_shape = self.distribution_1.batch_shape
        super(MultiCategoryDistribution, self).__init__(batch_shape, validate_args=validate_args)

    @property
    def probs_1(self):
        return self.distribution_1.probs
    
    @property
    def logits_1(self):
        return self.distribution_1.logits
    
    @property
    def probs_2(self):
        return self.distribution_2.probs 
    
    @property
    def logits_2(self):
        return self.distribution_2.logits

    def expand(self, batch_shape, _instance=None):
        raise NotADirectoryError
    
    def __new(self, probs_1, logits_1, probs_2, logits_2):
        self.distribution_1 = Categorical(probs_1, logits_1)
        self.distribution_2 = Categorical(probs_2, logits_2)
        self._param_1 = self.probs_1 if probs_1 is not None else self.logits_1
        self._param_2 = self.probs_2 if probs_2 is not None else self.logits_2

    def _new(self, *args, **kwargs):
        self.__new(*args, **kwargs)
        return self

    @constraints.dependent_property(is_discrete=True, event_dim=0)
    def support(self):
        raise NotImplementedError

    @lazy_property
    def logits(self):
        return (self.distribution_1.logits, self.distribution_2.logits)

    @lazy_property
    def probs(self):
       return self.distribution_1.probs, self.distribution_2.probs

    @property
    def param_shape(self):
        return (self._param_1.size(), self._param_2.size())

    @property
    def mean(self):
        return (self.distribution_1.mean, self.distribution_2.mean)

    @property
    def mode(self):
        return torch.cat((self.distribution_1.mode, self.distribution_2.mode), dim=-1) 
        # return (self.distribution_1.mode, self.distribution_2.mode)

    @property
    def variance(self):
        return (self.distribution_1.variance, self.distribution_2.variance)

    def sample(self, sample_shape=torch.Size()):
        # (batch, 1)
        s_1 = self.distribution_1.sample(sample_shape=sample_shape)
        s_1 = s_1[..., None]
        # (batch, 1)
        s_2 = self.distribution_2.sample(sample_shape=sample_shape)
        s_2 = s_2[..., None]
        return torch.cat((s_1, s_2), dim=-1) 
        
        
    def log_prob(self, value):
        s_1 = value[..., 0]
        s_2 = value[..., 1]
        log_prob_1 = self.distribution_1.log_prob(s_1)
        log_prob_2 = self.distribution_2.log_prob(s_2)
        return log_prob_1 + log_prob_2
    
    def entropy(self):
        entropy_1 = self.distribution_1.entropy()
        entropy_2 = self.distribution_2.entropy()
        return entropy_1 + entropy_2

    def enumerate_support(self, expand=True):
        raise NotImplementedError
    
 