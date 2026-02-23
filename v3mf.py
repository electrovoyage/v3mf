import re
import os
import math
import tqdm
import struct
import typing
import tempfile
import argparse
import srctools
import threading
import subprocess
import srctools.bsp
import srctools.mdl
import srctools.game
import srctools.filesys

from functools import lru_cache
from meshlib import mrmeshpy as mm

portalregex = re.compile(r'\([-0-9\.]+ [-0-9\.]+ [-0-9\.]+ \)')
numregex = re.compile(r'[-0-9\.]+')
vertexregex = re.compile(r'[-0-9\.]+ ([-0-9\.]+) ([-0-9\.]+) ([-0-9\.]+) ([-0-9\.]+) ([-0-9\.]+) ([-0-9\.]+)')

parser = argparse.ArgumentParser(__file__)

parser.add_argument('FILE')

parser.add_argument('--filetype', '-ft', help='what type is FILE', choices=['bsp', 'portal'])
parser.add_argument('--proptype', '-ap', help='entity classname to also render, with optional keyvalue name, formatted as classname.keyvalue. repeatable.', action='append')
parser.add_argument('--game', '-g', type=str, help='path to folder with game\'s main gameinfo.txt. must be specified if filetype=bsp and enableprops=true.')
parser.add_argument('--enableprops', '-ep', help='set to false to disable props in BSP mode', choices=['true', 'false'], default='true')
parser.add_argument('--crowbar', '-cr', type=str, help='path to CrowbarDecompiler.exe (https://github.com/mrglaster/Source-models-decompiler-cmd/releases/latest). required if props are enabled in BSP mode')

args = parser.parse_args()
final_mesh = mm.Mesh()

import open3d as o3d
import numpy as np

def read_smd(fp: str) -> tuple[tuple[tuple[float, float, float]], tuple[tuple[float, float, float]]]:
    with open(fp, 'r') as f:
        # skip until triangles
        while (line := f.readline()) and 'triangles' not in line:
            pass
        
        tris = []
        trinormals = []
        while (material_or_end := f.readline()) and material_or_end.strip() != 'end':
            vertices = []
            vertnormals = []
            for vertindex in range(3):
                vertdata = f.readline()
                
                x, y, z, nx, ny, nz = re.search(vertexregex, vertdata).groups()
                vertices.append((x, y, z))
                vertnormals.append((nx, ny, nz))
                
            tris.append(tuple(vertices))
            trinormals.append(tuple(vertnormals))
            
        return (tuple(tris), tuple(trinormals))
    

        
#read_smd("E:\p2 dump\models\props_gameplay\decomp\circuit_breaker_box\circuit_breaker_box_model.smd")ё

def is_plausible_count(value, max_reasonable=64):
    """Return True if value seems like a reasonable count (not negative, not huge)."""
    return 0 <= value <= max_reasonable

def read_vtx_and_build_mesh(vtx_filepath: str, vertices: list[tuple[float, float, float]]):
    with open(vtx_filepath, 'rb') as f:
        pass

def vvd_to_mesh_file(vertices: list[tuple[float, float, float]], normals: list[tuple[float, float, float]], vtx_filepath: str, lod: int=1, output_file: str="output.ply"):
    '''triangle_indices = vtx_to_triangles(vertices, vtx_filepath, lod=lod, has_extra_topology=False, verbose=True)

    # Convert to numpy arrays
    verts_np = np.array(vertices, dtype=np.float64)
    tris_np = np.array(triangle_indices, dtype=np.int32)

    # Create Open3D mesh and save
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_np)
    mesh.triangles = o3d.utility.Vector3iVector(tris_np)
    mesh.compute_vertex_normals()  # optional but helps visualization'''
    mesh = read_vtx_and_build_mesh(vtx_filepath, vertices)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])

    o3d.io.write_triangle_mesh(output_file, mesh)

def smd_to_mesh_file(smd_file_path, output_file="output.ply"):
    # Load SMD
    #smd = Smd(smd_file_path)
    #triangles_data = smd.triangles
    
    triangles_data = read_smd(smd_file_path)

    # Debug: print first triangle to verify structure

    # Extract unique vertices and triangle indices
    vertices = []
    triangle_indices = []
    vertex_map = {}
            
    for verts_in_triangle, normals in zip(*triangles_data):
        tri_verts = []
        for vert_def in verts_in_triangle:
            pos = vert_def
            pos_key = tuple(pos)
            if pos_key not in vertex_map:
                vertex_map[pos_key] = len(vertices)
                vertices.append(pos)
            tri_verts.append(vertex_map[pos_key])
        if len(tri_verts) == 3:
            triangle_indices.append(tri_verts)

    # Convert to numpy arrays
    verts_np = np.array(vertices, dtype=np.float64)
    tris_np = np.array(triangle_indices, dtype=np.int32)

    # Create Open3D mesh and save
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_np)
    mesh.triangles = o3d.utility.Vector3iVector(tris_np)
    mesh.compute_vertex_normals()  # optional but helps visualization

    o3d.io.write_triangle_mesh(output_file, mesh)
    #print(f"Mesh saved with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles.")

def readVVD(file: srctools.mdl.FileRSeek) -> tuple[list[tuple[float, float, float]]]:
    '''
    Read a VVD file from a Filesystem.
    
    :param file: File obtained from a Filesystem.
    :type file: srctools.mdl.FileRSeek
    :return: A 2-tuple containing a list of 3-tuples for vertex coordinates and a list of 3-tuples for vertex normals.
    :rtype: tuple[list[tuple[float, float, float]]]
    '''
    
    # read header
    (
        id,
        version,
        checksum,
        numLODs,
        *numLODVertices,
        numFixups,
        fixupTableStart,
        vertexDataStart,
        tangentDataStart
    ) = struct.unpack("4i 8i 4i", file.read(64))
    
    num_vertices: int = numLODVertices[0]
    
    file.seek(vertexDataStart, 0)
    
    vertices = []
    normals = []
    for i in range(num_vertices):
        #posx, posy, posz, normx, normy, normz = struct.unpack('16x 3f 3f 8x', file.read(48))
        file.read(16)
        position = struct.unpack('3f', file.read(12))
        normal = struct.unpack('3f', file.read(12))
        file.read(8)
        
        vertices.append(position)
        normals.append(normal)
    
    return (vertices, normals)

def addProp(prop: srctools.bsp.StaticProp, fs: srctools.filesys.FileSystemChain):
    with tempfile.TemporaryDirectory('-v3mf') as d:
        m = mm.Mesh(makeMeshFromSourceModel(prop.model, d, fs))
        
        angles = prop.angles
        
        m.transform(mm.AffineXf3f.linear(mm.Matrix3f.rotation(mm.Vector3f.plusX(), math.radians(angles.roll))))
        m.transform(mm.AffineXf3f.linear(mm.Matrix3f.rotation(mm.Vector3f.plusY(), math.radians(angles.pitch))))
        m.transform(mm.AffineXf3f.linear(mm.Matrix3f.rotation(mm.Vector3f.plusZ(), math.radians(angles.yaw))))
        m.transform(mm.AffineXf3f.translation(mm.Vector3f(*prop.origin.as_tuple())))
        final_mesh.addMesh(m)

@lru_cache
def makeMeshFromSourceModel(name: str, d: str, fs: srctools.filesys.FileSystemChain) -> dict[str, mm.Mesh | float]:
    #print('    extract files...')
    '''for ext in srctools.mdl.MDL_EXTS:
        newname = name.replace('.mdl', ext)
        try:
            with fs.open_bin(newname) as f:
                with open(os.path.join(d, os.path.basename(newname)), 'wb') as f2:
                    f2.write(f.read())
        except FileNotFoundError:
            pass
        
    #print('    decompile...')
    subprocess.run([args.crowbar, os.path.join(d, os.path.basename(name)), d], stdout=subprocess.PIPE, stderr=subprocess.PIPE).check_returncode()
    
    # now let's see if it saved as MODEL.smd or MODEL_reference.smd
    if os.path.exists(os.path.join(d, os.path.basename(name.replace('.mdl', '.smd')))):
        fpath = os.path.join(d, os.path.basename(name.replace('.mdl', '.smd')))
    elif os.path.exists(os.path.join(d, os.path.basename(name.replace('.mdl', '_reference.smd')))):
        fpath = os.path.join(d, os.path.basename(name.replace('.mdl', '_reference.smd')))
    elif os.path.exists(os.path.join(d, os.path.basename(name.replace('.mdl', '_model.smd')))):
        fpath = os.path.join(d, os.path.basename(name.replace('.mdl', '_model.smd')))
    # if there's still no good path, just find the first one that isn't an LOD or physics
    else:
        for file in os.listdir(d):
            fp = os.path.join(d, file)
            # TODO: this could fail for models that have "physics" in the actual model name
            if os.path.isfile(fp) and 'lod' not in file and 'physics' not in file and file.endswith('.smd'):
                fpath = os.path.join(d, file)
                
        # if still no results        
        if fpath is None:
            print(os.listdir(d))
            raise FileNotFoundError('failed to find decompiled SMD')

    #print('    convert to mesh...')
    smd_to_mesh_file(fpath, os.path.join(d, 'modelmesh.stl'))'''
    
    with open(os.path.join(d, 'vtx.vtx'), 'wb') as f:
        with fs.open_bin(name.replace('.mdl', '.vtx')) as f2:
            f.write(f2.read())
    
    vertices, normals = readVVD(fs.open_bin(name.replace('.mdl', '.vvd')))
    vvd_to_mesh_file(vertices, normals, os.path.join(d, 'vtx.vtx'), 1, os.path.join(d, 'modelmesh.stl'))
    
    #print('    add mesh to output...')
    m = mm.loadMesh(os.path.join(d, 'modelmesh.stl'))
    
    return m

if __name__ == '__main__':
    match args.filetype:
        case "bsp":
            bsp = srctools.bsp.BSP(args.FILE, srctools.bsp.VERSIONS.PORTAL_2)

            vert_blacklist: list[srctools.Vec] = []

            '''print("constructing base map brush")
            for entity, bmodel in bsp.bmodels.items():
                #print(entity['targetname'])
                for face in bmodel.faces:
                    cl = mm.PointCloud()
                    for edge in face.edges:
                        #cl.addPoint(mm.Vector3f(*edge.a.as_tuple()))
                        cl.addPoint(mm.Vector3f(*edge.b.as_tuple()))

                    mesh = mm.triangulatePointCloud(cl)
                    final_mesh.addMesh(mesh)
                break'''

            if args.enableprops != 'false':
                if getattr(args, 'game', None) is None:
                    raise Exception('must specify --game if props are enabled')

                if getattr(args, 'crowbar', None) is None:
                    raise Exception('must specify --crowbar if props are enabled')

                print("creating filesystem chain")
                game = srctools.game.Game(args.game)
                
                fs = game.get_filesystem()

                # mount BSP
                fs.add_sys(srctools.filesys.ZipFileSystem(args.FILE, zipfile=bsp.pakfile))

                print("populating static props")
                bar = tqdm.tqdm(bsp.props, unit=' props')
                for prop in bar:
                    #print(f"placing prop {prop.model} at {prop.origin}:")
                    #print('    create folder...')
                    addProp(prop, fs)
                #with multiprocessing.Pool(processes=4) as pool:
                #    pool.map(addProp, zip(bar, ([fs, ] * len(bar))))
                #WorkerPool(addProp, bar, fs, 4).run()
                    #for prop in bar:
                    #    pool.apply(addProp, [prop, fs])
                    #pool.apply(addProp, )

        case "portal":
            with open(args.FILE, 'r') as pfile:
                # first 3 lines are info we don't need
                pfile.readline()
                pfile.readline()
                pfile.readline()

                while (line := pfile.readline()):
                    corners_s = re.findall(portalregex, line)
                    corners = [[float(j) for j in re.findall(numregex, i)] for i in corners_s]

                    points = mm.std_vector_Vector3_float()

                    cl = mm.PointCloud()
                    for corner in corners:
                        cl.addPoint(mm.Vector3f(*corner))

                    #points.reverse()
                    #cl.points = mm.VertCoords(points)

                    # 100, 50, 180, 90, 100, True, None
                    nefertiti_mesh = mm.triangulatePointCloud(cl, mm.TriangulationParameters())
                    #mm.saveMesh(nefertiti_mesh, "Mesh.stl")
                    final_mesh.addMesh(nefertiti_mesh)

    mm.saveMesh(final_mesh, "Mesh.stl")