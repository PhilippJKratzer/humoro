import scipy.misc

def downsample(traj, factor=4):
    if factor <= 1:
        return
    traj.data = traj.data[::factor]
    traj.startframe = int(traj.startframe / factor)


def storeimg(pbul, name, campos=[-3, 3, 2], cam_auf=[0.5, 0.5, 1.]):
    width = 1920
    height = 1080

    fov = 60
    aspect = width / float(height)
    near = 0.02
    far = 10.

    view_matrix = pbul.computeViewMatrix(campos, cam_auf, [0, 0, 1])
    projection_matrix = pbul.computeProjectionMatrixFOV(fov, aspect, near, far)

    # Get depth values using the OpenGL renderer
    camimg = pbul.getCameraImage(width,
                                 height,
                                 view_matrix,
                                 projection_matrix,
                                 shadow=True,
                                 renderer=pbul.ER_BULLET_HARDWARE_OPENGL)
    scipy.misc.imsave(name, camimg[2])
