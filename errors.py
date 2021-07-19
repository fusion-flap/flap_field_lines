from builtins import Exception

class NoSurfaceFileError(Exception):
    def __init__(self, number=None):
        if number:
            self.message = f'No file found for surface number: {number}'
        else:
            self.message = 'Directory not found.'
        super().__init__(self.message)

class NoFsInfoError(Exception):
    def __init__(self):
        self.message = 'fs_info is not found at default location!'
        super().__init__(self.message)

class WrongConfigurationError(Exception):
    def __init__(self, config):
        self.message = f'{config} is not a valid magnetic confoguration!'
        super().__init__(self.message)

class WrongDirectionError(Exception):
    def __init__(self):
        self.message = f'Direction of field lines should be "forward", \
                        "backward" or "both"!'
        super().__init__(self.message)
