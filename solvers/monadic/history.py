import typing


class History:
    def __init__(self):
        self.init = None
        self.steps: typing.List = []
        self.final = None

    def clear(self):
        self.init, self.steps, self.final = None, [], None
