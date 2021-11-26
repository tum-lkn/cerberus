def factory_decorator(factory_dict, factory=None):
    def inner(cls):
        factory_dict[cls.__name__] = factory if factory is not None else cls

        return cls

    return inner
