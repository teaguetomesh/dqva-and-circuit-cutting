import abc

class ComponentInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        is_subclass = True
        functions = ['load_input', 'run', 'observe', 'close']
        for function in functions:
            is_subclass = is_subclass and hasattr(subclass, function) and callable(getattr(subclass, function))
        return is_subclass

    @abc.abstractmethod
    def run(self, input_content):
        '''Run the input_content'''
        raise NotImplementedError

    @abc.abstractmethod
    def get_output(self, options):
        '''Get the output of the component'''
        raise NotImplementedError

    @abc.abstractmethod
    def close(self, message: str):
        '''Shut down the component'''
        raise NotImplementedError