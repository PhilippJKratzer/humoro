# -*- coding: utf-8 -*-
# transformations.py

# Copyright (c) 2006, Christoph Gohlke
# Copyright (c) 2006-2009, The Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from __future__ import division

import math
import numpy as np
from numpy import linalg as la

# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())


class Affine3d:
    """
        3D dimensional affine/homogenous transform
        The rotation is encoded using quaternions.
        Uses the ROS convention for the quaternion (x, y, z, w)
        This should be the main interface for this module.

        TODO: should write a test for this class
    """
    translation = None
    rotation = None

    def __init__(self, t, r=np.array([0., 0., 0., 1.])):
        if t.shape == (4, 4):
            self._set_matrix(t)
        elif t.shape == (3, ):
            self.translation = t
            self.rotation = r

    def _set_matrix(self, M):
        if M.shape != (4, 4):
            raise ValueError('Matrix not of the correct size')
        self.translation = np.squeeze(np.asarray(M[0:3, 3]))
        self.rotation = quaternion_from_matrix(M)

    def linear(self):
        return np.mat(quaternion_matrix(self.rotation))

    def matrix(self):
        return np.bmat([[self.linear(),
                            np.matrix(self.translation).transpose()],
                           [np.mat([0., 0., 0., 1.])]])

    def __mul__(self, p):
        p_mat = np.mat(np.concatenate((p, [1]))).transpose()
        return np.squeeze(np.asarray(self.matrix() * p_mat)[:3])

    def __str__(self):
        ss = "Transform :\n"
        ss += " - translation (x = {:.4f}, y = {:.4f}, z = {:.4f})\n".format(
            self.translation[0],
            self.translation[1],
            self.translation[2])
        ss += " - rotation \
   (x = {:.4f}, y = {:.4f}, z = {:.4f}, w = {:.4f})\n".format(
            self.rotation[0], self.rotation[1],
            self.rotation[2], self.rotation[3])
        return ss


def quaternion_matrix(quaternion):
    """Return rotation matrix from quaternion.

    >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    >>> np.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
    True
    >>> M = quaternion_matrix([1, 0, 0, 0])
    >>> np.allclose(M, np.identity(4))
    True
    >>> M = quaternion_matrix([0, 1, 0, 0])
    >>> np.allclose(M, np.diag([1, -1, -1, 1]))
    True

    """
    # We assum the ROS convention (x, y, z, w)
    quaternion_tmp = np.array([0.0] * 4)
    quaternion_tmp[1] = quaternion[0]  # x
    quaternion_tmp[2] = quaternion[1]  # y
    quaternion_tmp[3] = quaternion[2]  # z
    quaternion_tmp[0] = quaternion[3]  # w
    q = np.array(quaternion_tmp, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
        [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]]])


def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.

    >>> q = quaternion_from_matrix(np.identity(4), True)
    >>> np.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(np.diag([1, -1, -1, 1]))
    >>> np.allclose(q, [0, 1, 0, 0]) or np.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> np.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> np.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> np.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> is_same_quaternion(quaternion_from_matrix(R, isprecise=False),
    ...                    quaternion_from_matrix(R, isprecise=True))
    True
    >>> R = euler_matrix(0.0, 0.0, np.pi/2.0)
    >>> is_same_quaternion(quaternion_from_matrix(R, isprecise=False),
    ...                    quaternion_from_matrix(R, isprecise=True))
    True

    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            q = q[[3, 0, 1, 2]]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                         [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                         [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                         [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)

    # We assume the ROS convention (x, y, z, w)
    quaternion_tmp = np.array([0.0] * 4)
    quaternion_tmp[3] = q[0]  # w
    quaternion_tmp[0] = q[1]  # x
    quaternion_tmp[1] = q[2]  # y
    quaternion_tmp[2] = q[3]  # z
    return quaternion_tmp


def euler_from_matrix(matrix, axes='sxyz'):

    # Return Euler angles from rotation matrix for specified axis sequence.
    #
    # axes : One of 24 axis sequences as string or encoded tuple
    #
    # Note that many Euler angle triplets can describe one matrix.
    #
    # >>> R0 = euler_matrix(1, 2, 3, 'syxz')
    # >>> al, be, ga = euler_from_matrix(R0, 'syxz')
    # >>> R1 = euler_matrix(al, be, ga, 'syxz')
    # >>> np.allclose(R0, R1)
    # True
    # >>> angles = (4.0*math.pi) * (np.random.random(3) - 0.5)
    # >>> for axes in _AXES2TUPLE.keys():
    # ...    R0 = euler_matrix(axes=axes, *angles)
    # ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
    # ...    if not np.allclose(R0, R1): print axes, "failed"

    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > _EPS:
            ax = math.atan2(M[i, j], M[i, k])
            ay = math.atan2(sy, M[i, i])
            az = math.atan2(M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(sy, M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if cy > _EPS:
            ax = math.atan2(M[k, j], M[k, k])
            ay = math.atan2(-M[k, i], cy)
            az = math.atan2(M[j, i], M[i, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(-M[k, i], cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return np.array([ax, ay, az])


def rtocarda(R, i, j, k):

    # RTOCARDA (Spacelib): Rotation  matrix  to Cardan or Eulerian angles.
    #
    # Extracts the Cardan (or Euler) angles from a rotation matrix.
    # The parameters  i, j, k  specify the sequence of the rotation axes
    # (their value must be the constant (X,Y or Z).
    # j must be different from i and k, k could be equal to i.
    # The two solutions are stored in the  three-element vectors q1 and q2.
    # RTOCARDA performs the inverse operation than CARDATOR.
    # Usage:
    #
    # [q1,q2]=rtocarda(R,i,j,k)
    #
    # Related functions : MTOCARDA
    #
    # (c) G.Legnani, C. Moiola 1998; adapted from: G.Legnani and R.Adamini 1993

    # spheader
    # disp('got this far')
    # if ( i<X | i>Z | j<X | j>Z | k<X | k>Z | i==j | j==k )
    # 	error('Error in RTOCARDA: Illegal rotation axis ')
    # end

    a = np.array([0.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0])

    # print "R : ", R

    if (j - i + 3) % 3 == 1:
        sig = 1  # ciclic
    else:
        sig = -1  # anti ciclic

    if i != k:  # Cardanic Convention

        i -= 1
        j -= 1
        k -= 1

        a[0] = math.atan2(-sig * R[j, k], R[k, k])
        a[1] = math.asin(sig * R[i, k])
        a[2] = math.atan2(-sig * R[i, j], R[i, i])

        b[0] = math.atan2(sig * R[j, k], -R[k, k])
        b[1] = ((math.pi - math.asin(sig * R[i, k]) + math.pi) %
                2 * math.pi) - math.pi
        b[2] = math.atan2(sig * R[i, j], -R[i, i])

    else:  # Euleriana Convention

        l = 6 - i - j  # noqa: E741

        i -= 1
        j -= 1
        k -= 1
        l -= 1  # noqa: E741

        a[0] = math.atan2(R[j, i], -sig * R[l, i])
        a[1] = math.acos(R[i, i])
        a[2] = math.atan2(R[i, j], sig * R[i, l])

        b[0] = math.atan2(-R[j, i], sig * R[l, i])
        b[1] = -math.acos(R[i, i])
        b[2] = math.atan2(-R[i, j], -sig * R[i, l])

    # report in degrees instead of radians
    a = a * 180 / math.pi
    b = b * 180 / math.pi

    # print "a : ", a
    # print "b : ", b

    return [a, b]


def normalize(x):

    y = np.matrix(np.eye(3))

    # y[0, :] = x[0, :] / la.norm(x[0, :])
    # y[1, :] = x[1, :] / la.norm(x[1, :])
    # y[2, :] = x[2, :] / la.norm(x[2, :])

    y[:, 0] = x[:, 0] / la.norm(x[:, 0])
    y[:, 1] = x[:, 1] / la.norm(x[:, 1])
    y[:, 2] = x[:, 2] / la.norm(x[:, 2])

    # important to export as matrix
    return np.matrix(y)


def quaternion_to_euler_angle(quat, wfront=True):
    if wfront:
        w = quat[0]
        x = quat[1]
        y = quat[2]
        z = quat[3]
    else:
        w = quat[3]
        x = quat[0]
        y = quat[1]
        z = quat[2]
        
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.atan2(t3, t4)

    return X, Y, Z

def quat_to_euler_2(quat):
    x = quat.T[0]
    y = quat.T[1]
    z = quat.T[2]
    w = quat.T[3]
    t0 = 2. * (w * x + y * z)
    t1 = 1. - 2. * (x * x + y * y)
    X = np.arctan2(t0, t1)
    t2 = 2.0 * (w * y - z * x)
    t2 = np.minimum(0.99999, t2)
    t2 = np.maximum(-0.99999, t2)
    Y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = np.arctan2(t3, t4)
    return np.stack([X, Y, Z]).T


def rotmat3d(angle, axis=0):
    """ rotation matrix around axis x,y,z """
    if type(angle) is np.ndarray:
        N = len(angle)
    else:
        N = 1
    c = np.cos(angle)
    s = np.sin(angle)
    res = np.zeros((N, 3, 3))
    res[:] = np.identity(3)
    if axis == 0:
        res[:, 1, 1] = c
        res[:, 2, 2] = c
        res[:, 2, 1] = s
        res[:, 1, 2] = -s
    elif axis == 1:
        res[:, 0, 0] = c
        res[:, 2, 2] = c
        res[:, 0, 2] = s
        res[:, 2, 0] = -s
    elif axis == 2:
        res[:, 0, 0] = c
        res[:, 1, 1] = c
        res[:, 1, 0] = s
        res[:, 0, 1] = -s
    if N == 1:
        return res[0]
    return res


def eulerToRotmat(angles):
    X = rotmat3d(angles[0], axis=0)
    Y = rotmat3d(angles[1], axis=1)
    Z = rotmat3d(angles[2], axis=2)
    return np.dot(Z, np.dot(Y, X))

def eulerToQuaternion(angles):
    return(quaternion_from_matrix(eulerToRotmat(angles)))

def rotmatAxis(angle, axis):
    """ computes a 3d rotation matrix from an angle around a 3d axis """
    c = np.cos(angle)
    s = np.sin(angle)
    ux = axis[0]
    uy = axis[1]
    uz = axis[2]
    res = np.array([[c + ux * ux * (1 - c), ux * uy * (1 - c) - uz * s, ux * uz * (1 - c) + uy * s],
                      [uy * ux * (1 - c) + uz * s, c + uy * uy * (1 - c), uy * uz * (1 - c) - ux * s],
                      [uz * ux * (1 - c) - uy * s, uz * uy * (1 - c) + ux * s, c + uz * uz * (1 - c)]])
    return res

def points2quat(v1, v2):
    """ computes a quaternion from the rotation between two vectors """
    q = np.zeros(4)
    a = np.cross(v1, v2);
    q[:3] = a;
    q[3] = np.sqrt((np.linalg.norm(v1) ** 2) * (np.linalg.norm(v2) ** 2)) + np.dot(v1, v2);
    return q


def quat2expmap(q):
  """
  Converts a quaternion to an exponential map
  Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/quat2expmap.m#L1

  Args
    q: 1x4 quaternion
  Returns
    r: 1x3 exponential map
  Raises
    ValueError if the l2 norm of the quaternion is not close to 1
  """
  if (np.abs(np.linalg.norm(q)-1)>1e-3):
    raise(ValueError, "quat2expmap: input quaternion is not norm 1")

  sinhalftheta = np.linalg.norm(q[:-1])
  coshalftheta = q[3]

  r0    = np.divide( q[:-1], (np.linalg.norm(q[:-1]) + np.finfo(np.float32).eps));
  theta = 2 * np.arctan2( sinhalftheta, coshalftheta )
  theta = np.mod( theta + 2*np.pi, 2*np.pi )

  if theta > np.pi:
    theta =  2 * np.pi - theta
    r0    = -r0

  r = r0 * theta
  return r


def expmap2rotmat(r):
  """
  Converts an exponential map angle to a rotation matrix
  Matlab port to python for evaluation purposes
  I believe this is also called Rodrigues' formula
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/expmap2rotmat.m

  Args
    r: 1x3 exponential map
  Returns
    R: 3x3 rotation matrix
  """
  theta = np.linalg.norm( r )
  r0  = np.divide( r, theta + np.finfo(np.float32).eps )
  r0x = np.array([0, -r0[2], r0[1], 0, 0, -r0[0], 0, 0, 0]).reshape(3,3)
  r0x = r0x - r0x.T
  R = np.eye(3,3) + np.sin(theta)*r0x + (1-np.cos(theta))*(r0x).dot(r0x);
  return R

def expmap_to_quat(expmap):
    expmap = np.array(expmap)
    theta = np.linalg.norm(expmap.T, axis=0)
    w = np.expand_dims(np.cos(.5*theta), 0)
    xyz = .5 * np.sin(.5*theta)/(.5*theta)*expmap.T
    return np.concatenate([xyz, w]).T

def quaternion_divide(a, b):
    bnorm = np.linalg.norm(b)
    x1, y1, z1, w1 = a[0], a[1], a[2], a[3]
    x2, y2, z2, w2 = b[0], b[1], b[2], b[3]
    w = (w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2) / bnorm
    x = (-w1 * x2 + x1 * w2 - y1 * z2 + z1 * y2) / bnorm
    y = (-w1 * y2 + x1 * z2 + y1 * w2 - z1 * x2) / bnorm
    z = (-w1 * z2 - x1 * y2 + y1 * x2 + z1 * w2) / bnorm
    return [x, y, z, w]
