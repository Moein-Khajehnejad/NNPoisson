from sys import platform

class Paths:

    @staticmethod
    def data_path():
        if platform == "darwin":
            base_path = "../nongit/"
        elif platform == "linux":
            base_path = "/datasets/work/mlaifsp-dm/work/neuro-pixel/"
        else:
            raise Exception('unknown platform')
        return base_path

    @staticmethod
    def output_path():
        if platform == 'darwin':
            base_path = "../nongit/"
        elif platform == 'linux':
            base_path = "/scratch1/dez004/deep_neuro/nongit/"
        else:
            raise Exception('unknown platform')
        return base_path

