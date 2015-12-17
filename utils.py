try:
    xrange
except NameError:
    xrange = range

def InputCell(object):
    def __init__(self, **entries): 
        self.__dict__.update(entries)

def OutputCell(object):
    def __init__(self, **entries): 
        self.__dict__.update(entries)

