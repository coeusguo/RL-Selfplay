import abc

class Agent(abc.ABC):

    @abc.abstractclassmethod
    def init_buffer(self):
        '''
        generic methods, will be called by rollout sampler to clean
        the buffer (if any), after loading the new policy checkpoint
        '''
        pass

    @abc.abstractclassmethod
    def run_one_episode(self):
        pass

