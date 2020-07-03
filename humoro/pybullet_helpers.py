def create_inv_index(human, p):
    inv_index = {}
    for j in range(p.getNumJoints(human)):
        info = p.getJointInfo(human, j)
        inv_index[info[1].decode("utf-8")] = info[0]
    return inv_index
