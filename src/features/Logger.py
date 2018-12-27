import time


class Logger:
    def __init__(self):
        self.time_pool = [time.time()]
        self.total_time = 0
        self.name = None
        pass

    def header(self):
        print('\n\n---------------------- '+self.name+' ----------------------\n')

    def sec_start(self, sec_name):
        print('\t[...] '+sec_name+'...', end='')
        self.time_pool.append(time.time())

    def sec_end(self):
        sec_start_time = self.time_pool.pop(len(self.time_pool)-1)
        print('\t\t  Finished in %.3f s!' % (time.time() - sec_start_time))

    def footer(self):
        class_start_time = self.time_pool.pop(0)
        print('--------- '+self.name+' Finished In [%.1f] Seconds ---------' % (time.time()-class_start_time))


class FullLogger(Logger):
    def footer(self):
        print()
        super().footer()
        print('\n')


class BriefLogger(Logger):
    def header(self):
        pass

    def sec_start(self, sec_name):
        pass

    def sec_end(self):
        pass


class SilentLogger(BriefLogger):
    def footer(self):
        pass


def create_logger(mode='full'):
    if mode == 'full':
        return FullLogger()
    elif mode == 'brief':
        return BriefLogger()
    elif mode == 'silent':
        return SilentLogger()
