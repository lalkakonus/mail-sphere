class Logging():
    def __init__(self, log_mode='INFO'):
        self.log_mode = log_mode
        self.mode_set = {'INFO' : ['INFO'],
                         'DEBUG' : ['DEBUG'],
                         'INFO + DEBUG' : ['INFO' , 'DEBUG']}
        
    def set_mode(self, mode):
        if mode in self.mode_set:
            self.log_mode = mode

    def log(self, text, *args, mode='INFO'):
        if self.log_mode == mode:
            if args is not None:
                print('[' + mode + '] ' + text.format(*args))
            else:
                print(text)
