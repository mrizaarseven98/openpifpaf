import os

import numpy as np
try:
    import matplotlib.cm as mplcm
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    pass

import openpifpaf


ROBOT_KEYPOINTS_3 = [
    'Robot_left',       # 1
    'Robot_right',        # 2
    'Robot_front'
    #'Robot_upfront'    # 3
    
]
ROBOT_SKELETON_3 = [(1,2), (2,3), (3,1)]

ROBOT_SCORE_WEIGHTS_3 = [0.3] * 3

ROBOT_SIGMAS_3 = [0.05] * 3

HFLIP_3 = {
    #'Robot_left':'Robot_right',
    #'Robot_front':'Robot_upfront'
}

ROBOT_CATEGORIES_3 = ["Mori"]

assert len(ROBOT_SCORE_WEIGHTS_3) == len(ROBOT_KEYPOINTS_3)


HEIGHT=1.2
# CAR POSE is used for joint rescaling. x = [-3, 3] y = [0,4]
ROBOT_POSE_3 = np.array([
    [-4.4, 1.5, HEIGHT],  # 'HEIGHT_up_right',              # 1
    [5, 1.5, HEIGHT],   # 'HEIGHT_up_left',               # 2
    [0.25, 9.7, HEIGHT],
    #[0.25, 5, HEIGHT]  # 'HEIGHT_light_right',           # 3
])



def get_constants():
    return [ROBOT_KEYPOINTS_3, ROBOT_SKELETON_3, HFLIP_3, ROBOT_SIGMAS_3, ROBOT_POSE_3, ROBOT_CATEGORIES_3, ROBOT_SCORE_WEIGHTS_3]


def draw_ann(ann, *, keypoint_painter, filename=None, margin=0.5, aspect=None, **kwargs):
    from openpifpaf import show  # pylint: disable=import-outside-toplevel

    bbox = ann.bbox()
    xlim = bbox[0] - margin, bbox[0] + bbox[2] + margin
    ylim = bbox[1] - margin, bbox[1] + bbox[3] + margin
    if aspect == 'equal':
        fig_w = 5.0
    else:
        fig_w = 5.0 / (ylim[1] - ylim[0]) * (xlim[1] - xlim[0])

    with show.canvas(filename, figsize=(fig_w, 5), nomargin=True, **kwargs) as ax:
        ax.set_axis_off()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        if aspect is not None:
            ax.set_aspect(aspect)

        keypoint_painter.annotation(ax, ann)


def draw_skeletons(pose, sigmas, skel, kps, scr_weights):
    from openpifpaf.annotation import Annotation  # pylint: disable=import-outside-toplevel
    from openpifpaf import show  # pylint: disable=import-outside-toplevel

    scale = np.sqrt(
        (np.max(pose[:, 0]) - np.min(pose[:, 0]))
        * (np.max(pose[:, 1]) - np.min(pose[:, 1]))
    )

    show.KeypointPainter.show_joint_scales = True
    keypoint_painter = show.KeypointPainter()
    ann = Annotation(keypoints=kps, skeleton=skel, score_weights=scr_weights)
    ann.set(pose, np.array(sigmas) * scale)
    os.makedirs('docs', exist_ok=True)
    draw_ann(ann, filename='docs/modular_robot.png', keypoint_painter=keypoint_painter)


def plot3d_red(ax_2D, p3d, skeleton):
    skeleton = [(bone[0] - 1, bone[1] - 1) for bone in skeleton]

    rot_p90_x = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    p3d = p3d @ rot_p90_x

    fig = ax_2D.get_figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.set_axis_off()
    ax_2D.set_axis_off()

    ax.view_init(azim=-90, elev=20)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    max_range = np.array([p3d[:, 0].max() - p3d[:, 0].min(),
                          p3d[:, 1].max() - p3d[:, 1].min(),
                          p3d[:, 2].max() - p3d[:, 2].min()]).max() / 2.0
    mid_x = (p3d[:, 0].max() + p3d[:, 0].min()) * 0.5
    mid_y = (p3d[:, 1].max() + p3d[:, 1].min()) * 0.5
    mid_z = (p3d[:, 2].max() + p3d[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)  # pylint: disable=no-member

    for ci, bone in enumerate(skeleton):
        c = mplcm.get_cmap('tab20')((ci % 20 + 0.05) / 20)  # Same coloring as Pifpaf preds
        ax.plot(p3d[bone, 0], p3d[bone, 1], p3d[bone, 2], color=c)

    def animate(i):
        ax.view_init(elev=10., azim=i)
        return fig

    return FuncAnimation(fig, animate, frames=360, interval=100)


def print_associations():
    print("\nAssociations of the ROBOT skeleton with 3 keypoints")
    for j1, j2 in ROBOT_SKELETON_3:
        print(ROBOT_KEYPOINTS_3[j1 - 1], '-', ROBOT_KEYPOINTS_3[j2 - 1])


def main():
    print_associations()
# =============================================================================
#     draw_skeletons(CAR_POSE_24, sigmas = CAR_SIGMAS_24, skel = CAR_SKELETON_24,
#                    kps = CAR_KEYPOINTS_24, scr_weights = CAR_SCORE_WEIGHTS_24)
#     draw_skeletons(CAR_POSE_66, sigmas = CAR_SIGMAS_66, skel = CAR_SKELETON_66,
#                    kps = CAR_KEYPOINTS_66, scr_weights = CAR_SCORE_WEIGHTS_66)
# =============================================================================
    with openpifpaf.show.Canvas.blank(nomargin=True) as ax_2D:
        anim_robot = plot3d_red(ax_2D, ROBOT_POSE_3, ROBOT_SKELETON_3)
    
        anim_robot.save('/home/riza/.local/lib/python3.8/site-packages/openpifpaf/plugins/modular_robot/docs/modular_robot_3D_skeleton.gif', fps=30)

if __name__ == '__main__':
    main()
