import matplotlib.pyplot as plt
import matplotlib.animation as animation



class MoviePlot:
    def __init__(self,figs={1:'pyplot_movie'},dpi=100,fps=15):
        self.writers=[]
        for fignum,filename in figs.items():
            fig = plt.figure(fignum)
            print("Record figure {} as {}".format(fignum,filename))
            FFMpegWriter = animation.writers['ffmpeg']
            metadata = dict(title='Movie Test', artist='Matplotlib',
                            comment='Movie support!')
            writer = FFMpegWriter(fps=fps, metadata=metadata)
            writer.setup(fig, "{}.mp4".format(filename), dpi)
            self.writers.append(writer)

    def grab_frames(self):
        for w in self.writers:
            w.grab_frame()