import numpy as np
from config import Config
from movieplot import MoviePlot

movie = None

def display_progress(replay_buffer, flags, plt, RewardsHistory, Rdfr, env, episode, episodes, i_episode, actor, actorp, critic, criticp):
    QAccHistory = []
    global movie

    if flags.movie and not movie:
        m={1:"episode",2:"trends",3:''}
        if flags.viz:
            m.update({3:'critic',4:'actor'})
        movie=MoviePlot(m)
    if not flags.movie and movie:
        movie.finish()
        movie=None
    fig = plt.figure(1)
    sp = (4, 1)
    plt.clf()

    plt.subplot(*sp, 1)
    plt.subplots_adjust(left=0.2)
    # plt.gca().set_ylim([-1.2,1.2])
    plt.gca().axhline(y=0, color='k')
    fig.suptitle("{}, Episode {} {}{}".format(env.spec.id, i_episode, "Warming" if (i_episode < Config.warmup) else "",
                                              "/W noise" if flags.noise else ""))
    for i in range(replay_buffer.obs[episode].shape[1]):
        plt.plot(replay_buffer.obs[episode, i], label='obs {}'.format(i))
    plt.legend(loc=1)
    plt.subplot(*sp, 2)
    plt.gca().axhline(y=0, color='k')
    plt.plot(replay_buffer.action[episode], 'g', label='action taken')
    actionp = actorp.predict(replay_buffer.obs[episode])
    action = actor.predict(replay_buffer.obs[episode])
    plt.plot(action, 'red', label='action')
    plt.plot(actionp, 'lightgreen', label='actionp')
    plt.legend(loc=1)
    plt.subplot(*sp, 3)
    plt.gca().axhline(y=0, color='k')
    plt.plot(replay_buffer.reward[episode], 'r', label='reward')
    plt.legend(loc=1)
    plt.subplot(*sp, 4)
    plt.gca().axhline(y=0, color='k')
    q = critic.predict([replay_buffer.obs[episode], replay_buffer.action[episode]])
    plt.plot(q, 'k', label='Q')
    qp = criticp.predict([replay_buffer.obs[episode], replay_buffer.action[episode]])
    plt.plot(qp, 'gray', label='Qp')
    Rdfr[episode] = replay_buffer.reward[episode]
    last = 0
    for i in reversed(episode):
        Rdfr[i] += Config.gamma * last
        last = Rdfr[i]

    plt.plot(Rdfr[episode], 'r', label='R')
    QAccHistory.append(np.mean(np.abs(Rdfr[episode] - qp)))
    plt.legend(loc=1)

    # simulation control widgets
    ax = plt.axes([0.01, 0.01, 0.1, 0.2])
    flags.showat(ax)

    # second plot
    plt.figure(2)
    sp = (2, 1)
    plt.clf()
    fig.suptitle("{}, Trends {}{}".format(env.spec.id,  "Warming" if (i_episode < Config.warmup) else "",
                                              "/W noise" if flags.noise else ""))
    plt.subplot(*sp, 1)
    plt.gca().axhline(y=0, color='k')
    plt.plot(RewardsHistory, 'r', label='reward history')
    plt.legend(loc=2)
    plt.subplot(*sp, 2)
    plt.gca().axhline(y=0, color='k')
    plt.plot(QAccHistory, 'r', label='Qloss history')
    plt.legend(loc=2)

    # third plot
    if flags.viz:
        fig = plt.figure(3)
        ax = plt.gca()
        plt.clf()
        ax.set_title("Qvalue for obs{}".format(Config.viz_idx))
        # todo: make this a function of the first two action space dimensions
        gsz = 100
        oidx0 = Config.viz_idx[0]
        oidx1 = Config.viz_idx[1]
        ndim = env.observation_space.shape[0]
        nadim = env.action_space.shape[0]
        low = env.observation_space.low
        high = env.observation_space.high
        extent = [low[oidx0], high[oidx0], low[oidx1], high[oidx1]]
        X, Y = np.meshgrid(np.linspace(high[oidx0], low[oidx0], gsz),
                           np.linspace(high[oidx1], low[oidx1], gsz))
        tmp = []
        for idx in range(ndim):
            if idx in Config.viz_idx:
                tmp.append(X if idx == Config.viz_idx[0] else Y)
            else:
                tmp.append(np.ones_like(X) * replay_buffer.obs[episode[0], idx])
        obs = np.array(tmp).T.reshape((gsz * gsz, ndim))
        act = actor.predict(obs)
        A = act.reshape(gsz, gsz, nadim)
        # print("act shape={} A={} act={}, A={}".format(act.shape,A.shape,act[:5],A[1,1]))
        Z = critic.predict([obs, act]).reshape(gsz, gsz)
        vmin = np.min(Z)
        vmax = np.max(Z)
        im = plt.imshow(Z, cmap=plt.cm.RdBu_r, vmin=vmin, vmax=vmax, extent=extent)
        im.set_interpolation('bilinear')
        cb = fig.colorbar(im)
        tail = int(1 / (1 - Config.gamma))
        plt.axis([low[0], high[0], low[1], high[1]])
        if flags.tails:
            mask = np.array([True] * ndim)
            mask[Config.viz_idx] = False
            for i, e in enumerate(episodes):
                # check if rest of episode observations match (i.e. same slice of Q)
                if np.any(np.logical_and(mask, (replay_buffer.obs[episodes[-1][0]] != replay_buffer.obs[e][0]))):
                    continue
                plt.scatter(x=-replay_buffer.obs[e[:-tail], 1], y=replay_buffer.obs[e[:-tail], 0], cmap=plt.cm.RdBu_r, c=Rdfr[e[:-tail]],
                            vmin=vmin, vmax=vmax, s=3)
        plt.scatter(x=-replay_buffer.obs[episodes[-1][:-tail], 1], y=replay_buffer.obs[episodes[-1][:-tail], 0], cmap=plt.cm.RdBu_r, c='k',
                    vmin=vmin, vmax=vmax, s=0.5)
        plt.scatter(x=-replay_buffer.obs[episodes[-1][-1], 1], y=replay_buffer.obs[episodes[-1][-1], 0], c='green', s=6)


        fig = plt.figure(4)
        ax = plt.gca()
        plt.clf()
        ax.set_title("Actions for obs{}".format(Config.viz_idx))
        plt.axis([low[0], high[0], low[1], high[1]])
        sp = (1,nadim)
        for i in range(nadim):
            plt.subplot(*sp, i + 1)
            avmin = env.observation_space.low[i]
            avmax = env.observation_space.high[i]
            im = plt.imshow(A[:, :, i], cmap=plt.cm.RdBu_r, vmin=avmin, vmax=avmax, extent=extent)
            im.set_interpolation('bilinear')
            plt.scatter(x=-replay_buffer.obs[episodes[-1], 1], y=replay_buffer.obs[episodes[-1], 0], c='k',
                                vmin=avmin, vmax=avmax, s=0.5)
            plt.scatter(x=-replay_buffer.obs[episodes[-1][-1], 1], y=replay_buffer.obs[episodes[-1][-1], 0], c='green', s=6)
        if movie:
            movie.grab_frames()
    plt.pause(0.1)
