from builtins import Exception

class NoSurfaceFileError(Exception):
    def __init__(self, number):
        self.message = f'No file found for surface number: {number}\n'
        super().__init__(self.message)