

def props(object):
    return [s for s in dir(object) if not s.startswith('_')]

def props_(object):
    return [s for s in dir(object) if not s.startswith('__')]