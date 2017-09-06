class Config(object):
        """These changes to python pep8 naming convention."""
        buffer_length = 1000
        batch_size = 16
        tau = 0.0001
        gamma = 0.95
        warmup = 2
        render_flag = False
        noise_flag = True
        viz_flag = True
        viz_idx = [0, 1]
        show_progress = True
        save_model = False
        max_episodes = 200000
        max_steps = 2500
