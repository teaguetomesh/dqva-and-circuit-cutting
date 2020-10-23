import abc

class ComponentInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        is_subclass = True
        functions = ['load_input', 'run', 'observe', 'close']
        for function in functions:
            is_subclass = is_subclass and hasattr(subclass, function) and callable(getattr(subclass, function))
        is_subclass = is_subclass or NotImplemented
        return is_subclass

    @abc.abstractmethod
    def load_input(self, component_input):
        '''Load the input'''
        raise NotImplementedError

    @abc.abstractmethod
    def run(self, options: dict):
        '''Run and return output'''
        raise NotImplementedError

    @abc.abstractmethod
    def observe(self):
        '''Observe the state of the component'''
        raise NotImplementedError

    @abc.abstractmethod
    def close(self, message: str):
        '''Shut down the component'''
        raise NotImplementedError