class Config(object):
    def __init__(self):
        """These changes to python pep8 naming convention."""
        self.buffer_length = 200000
        self.batch_size = 320
        self.tau = 0.01
        self.gamma = 0.5
        self.warmup = 2
        self.render_flag = False
        self.noise_flag = True
        self.viz_flag = True
        self.viz_idx = [0, 1]
        self.show_progress = True
        self.save_model = False

Rsz = 200000 #replay buffer size
N = 320 # batch size
tau = 0.001
gamma = .95
warmup = 2
renderFlag = False
noiseFlag = True
vizFlag = True
vizIdx = [0, 1]
showProgress = True
saveModel = False