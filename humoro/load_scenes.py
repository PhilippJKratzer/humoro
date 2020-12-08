import os
import sys

from humoro.objectParser import parseHDF5
import humoro.utility as util

def spawn_from_file(p, obj_ids, filename_xml):
    import xml.etree.ElementTree as ET
    root = ET.parse(filename_xml).getroot()
    obj_names = [""] * len(obj_ids)
    for child in root:
        if child.tag == "body":
            id = int(child.attrib['id'])
            if not id in obj_ids:
                continue
            name = child.attrib['name']
            obj_names[obj_ids.index(id)] = name
            for p_body in child:
                if p_body.tag == "meshfile":
                    meshfile = p_body.attrib["src"]
                elif p_body.tag == "color":
                    splits = p_body.text.split()
                    color = [float(splits[1]), float(splits[2]), float(splits[3]), float(splits[0])]
            if name == "goggles":
                pass
                #p.spawnObject(name, meshfile=None, color=[0,0,1,1])
            else:
                p.spawnObject(name, meshfile=os.path.join(os.path.dirname(os.path.dirname(filename_xml)), meshfile), color=color)
        else:
            print ("tag " + child.tag + " not understood")
    return obj_names


def autoload_objects(p, filename, scenefile, downsample=False):
    obj_trajs, obj_ids = parseHDF5(filename)
    obj_names = spawn_from_file(p, obj_ids, scenefile)
    if downsample:
        for i in range(len(obj_trajs)):
            util.downsample(obj_trajs[i])
    p.addPlaybackTrajObjList(obj_trajs, obj_names)
    return obj_trajs, obj_names
