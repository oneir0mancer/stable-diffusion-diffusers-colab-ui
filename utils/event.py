from collections.abc import Callable

class Event:
    def __init__(self):
        self.__callbacks = []

    def add_callback(self, callback: Callable):
        self.__callbacks.append(callback)

    def remove_callback(self, callback: Callable):
        self.__callbacks.remove(callback)

    def clear_callbacks(self):
        self.__callbacks.clear()
    
    def invoke(self, event_args: list = None):
        if event_args is None:
            for x in self.__callbacks: x()
        else:
            for x in self.__callbacks: x(event_args)

    @property
    def callbacks(self): return self.__callbacks
