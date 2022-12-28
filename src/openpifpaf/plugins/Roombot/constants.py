import os

import numpy as np
try:
    import matplotlib.cm as mplcm
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    pass

import openpifpaf


ROBOT_KEYPOINTS_50 = [
 'Robot_up1',
 'Robot_right1',
 'Robot_left1',
 'Robot_down1',
 'Robot_center1',
 'Robot_up2',
 'Robot_right2',
 'Robot_left2',
 'Robot_down2',
 'Robot_center2',
 'Robot_up3',
 'Robot_right3',
 'Robot_left3',
 'Robot_down3',
 'Robot_center3',
 'Robot_up4',
 'Robot_right4',
 'Robot_left4',
 'Robot_down4',
 'Robot_center4',
 'Robot_up5',
 'Robot_right5',
 'Robot_left5',
 'Robot_down5',
 'Robot_center5',
 'Robot_up6',
 'Robot_right6',
 'Robot_left6',
 'Robot_down6',
 'Robot_center6',
 'Robot_up7',
 'Robot_right7',
 'Robot_left7',
 'Robot_down7',
 'Robot_center7',
 'Robot_up8',
 'Robot_right8',
 'Robot_left8',
 'Robot_down8',
 'Robot_center8',
 'Robot_up9',
 'Robot_right9',
 'Robot_left9',
 'Robot_down9',
 'Robot_center9',
 'Robot_up10',
 'Robot_right10',
 'Robot_left10',
 'Robot_down10',
 'Robot_center10',

    
]
ROBOT_SKELETON_50 = [(1,2), (2,3), (3,4), (4,1), (1,5), (6,7), (7,8), (8,9), (9,6), (6,10), (11,12), (12,13), (13,14), (14,11), (11,15), (16,17), (17,18), (18,19), (19,16), (16,20),
(21,22), (22,23), (23,24), (24,21), (21,25), (26,27), (27,28), (28,29), (29,26), (26,30), (31,32), (32,33), (33,34), (34,31), (31,35), (36,37), (37,38), (38,39), (39,36), (36,40), (41,42), (42,43), (43,44), (44,41), (41,45),
(46,47), (47,48), (48,49), (49,46), (46,50)]

ROBOT_SCORE_WEIGHTS_50 = [0.3] * 50

ROBOT_SIGMAS_50 = [0.05] * 50

HFLIP_50 = {
    #'Robot_left':'Robot_right',
    #'Robot_front':'Robot_upfront'
}

ROBOT_CATEGORIES_50 = ["Roombot"]

assert len(ROBOT_SCORE_WEIGHTS_50) == len(ROBOT_KEYPOINTS_50)


HEIGHT=11
OFFSET_1=3
OFFSET_2=0.7
# CAR POSE is used for joint rescaling. x = [-3, 3] y = [0,4]
ROBOT_POSE_50 = np.array([
    [-0.7, 3, HEIGHT],  
    [3, 0.7, HEIGHT],
    [-3, -0.7, HEIGHT],  
    [0.7, -3, HEIGHT],   
    [0, 0, HEIGHT],

    [-0.7+11, 3, HEIGHT],  # 'HEIGHT_up_right',              # 1
    [3+11, 0.7, HEIGHT],
    [-3+11, -0.7, HEIGHT],  # 'HEIGHT_up_right',              # 1
    [0.7+11, -3, HEIGHT],   # 'HEIGHT_up_left',               # 2
    [0+11, 0, HEIGHT],

    [-0.7, -5.5, 8.5],  # 'HEIGHT_up_right',              # 1
    [3, -5.5, 6.2],
    [-3, -5.5, 4.8],  # 'HEIGHT_up_right',              # 1
    [0.7, -5.5, 2.5],   # 'HEIGHT_up_left',               # 2
    [0, -5.5, 5.5],

    [-0.7+11, -5.5, 8.5],  
    [3+11, -5.5, 6.2],
    [-3+11, -5.5, 4.8],  
    [0.7+11, -5.5, 2.5],   
    [0+11, -5.5, 5.5],

    [-5.5, 0.7, 8.5],  
    [-5.5, -3, 6.2],
    [-5.5, 3, 4.8],  
    [-5.5, -0.7, 2.5],   
    [-5.5, 0, 5.5],

    [0.7, 5.5, 8.5],  # 'HEIGHT_up_right',              # 1
    [-3, 5.5, 6.2],
    [3, 5.5, 4.8],  # 'HEIGHT_up_right',              # 1
    [-0.7, 5.5, 2.5],   # 'HEIGHT_up_left',               # 2
    [0, 5.5, 5.5],
    
    [0.7+11, 5.5, 8.5],  # 'HEIGHT_up_right',              # 1
    [-3+11, 5.5, 6.2],
    [3+11, 5.5, 4.8],  # 'HEIGHT_up_right',              # 1
    [-0.7+11, 5.5, 2.5],   # 'HEIGHT_up_left',               # 2
    [0+11, 5.5, 5.5],

    [-5.5+22, -0.7, 8.5],  
    [-5.5+22, 3, 6.2],
    [-5.5+22, -3, 4.8],  
    [-5.5+22, 0.7, 2.5],   
    [-5.5+22, 0, 5.5],

    [0.7, -3, 0],  
    [3, -0.7, 0],
    [-3, 0.7, 0],  
    [0.7, -3, 0],   
    [0, 0, 0],

    [0.7+11, -3, 0],  
    [3+11, -0.7, 0],
    [-3+11, 0.7, 0],  
    [0.7+11, -3, 0],   
    [0+11, 0, 0],
    

])



def get_constants():
    return [ROBOT_KEYPOINTS_50, ROBOT_SKELETON_50, HFLIP_50, ROBOT_SIGMAS_50, ROBOT_POSE_50, ROBOT_CATEGORIES_50, ROBOT_SCORE_WEIGHTS_50]


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
    draw_ann(ann, filename='docs/Roombot.png', keypoint_painter=keypoint_painter)


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
    for j1, j2 in ROBOT_SKELETON_50:
        print(ROBOT_KEYPOINTS_50[j1 - 1], '-', ROBOT_KEYPOINTS_50[j2 - 1])


def main():
    print_associations()
# =============================================================================
#     draw_skeletons(CAR_POSE_24, sigmas = CAR_SIGMAS_24, skel = CAR_SKELETON_24,
#                    kps = CAR_KEYPOINTS_24, scr_weights = CAR_SCORE_WEIGHTS_24)
#     draw_skeletons(CAR_POSE_66, sigmas = CAR_SIGMAS_66, skel = CAR_SKELETON_66,
#                    kps = CAR_KEYPOINTS_66, scr_weights = CAR_SCORE_WEIGHTS_66)
# =============================================================================
    with openpifpaf.show.Canvas.blank(nomargin=True) as ax_2D:
        anim_robot = plot3d_red(ax_2D, ROBOT_POSE_50, ROBOT_SKELETON_50)
    
        anim_robot.save('/home/riza/.local/lib/python3.8/site-packages/openpifpaf/plugins/Roombot/docs/Roombot_3D_skeleton.gif', fps=30)

if __name__ == '__main__':
    main()
