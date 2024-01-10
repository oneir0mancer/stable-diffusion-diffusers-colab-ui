class EmptyOutput:
    """An empty wrapper around output, for when Output widget wasn't provided"""
    def __enter__(self): pass

    def __exit__(self, type, value, traceback): pass

    def clear_output(self): pass