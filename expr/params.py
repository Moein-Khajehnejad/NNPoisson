class GetInps:
    model = None
    opt = None
    stim_size = None
    @staticmethod
    def get_stim_size():
        if GetInps.stim_size is None:
            raise ValueError("Stim size cannot be none")
        return GetInps.stim_size
