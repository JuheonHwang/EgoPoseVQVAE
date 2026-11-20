"""Skeleton plotting used by the test scripts."""
import matplotlib.pyplot as plt

__all__ = ['KINEMATIC_CHAINS', 'show_poses']

# Kinematic chains (joint ids) per skeleton key of utils.config.SKELETONS.
KINEMATIC_CHAINS = {
    '14': [[1, 0], [1, 2, 3, 4], [1, 5, 6, 7], [1, 8, 9, 10], [1, 11, 12, 13]],
    '16': [[1, 0], [1, 2, 3, 4], [1, 5, 6, 7], [1, 8, 9, 10, 11], [1, 12, 13, 14, 15]],
    '22': [[1, 2, 3, 4, 5, 0], [5, 6, 7, 8, 9], [5, 10, 11, 12, 13],
           [1, 14, 15, 16, 17], [1, 18, 19, 20, 21]],
}

COLORS = ['black', 'red', 'blue', 'red', 'blue']


def show_poses(poses, labels, skeleton, title):
    """Plot one 3D skeleton per panel.

    Arguments:
        poses {list} -- arrays of shape (1, J, 3); only the first sample of
            each (batch size 1) is drawn
        labels {list} -- x-axis label of each panel, e.g. ['Pred', 'GT']
        skeleton {str} -- key of KINEMATIC_CHAINS (``config.skeleton``)
        title {str} -- figure title
    """

    fig = plt.figure(figsize=plt.figaspect(0.5))
    fig.suptitle(title, fontsize=16)

    for k, (pose, label) in enumerate(zip(poses, labels)):
        pose = pose[0]
        ax = fig.add_subplot(1, len(poses), k + 1, projection='3d')
        for chain, color in zip(KINEMATIC_CHAINS[skeleton], COLORS):
            ax.plot3D(pose[chain, 0], pose[chain, 1], pose[chain, 2],
                      linewidth=4, color=color)
        ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2])
        ax.set_xlabel(label, fontsize=10, rotation=100)

    plt.show()
    plt.close(fig)
