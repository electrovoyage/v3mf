import re
import os
import math
import tqdm
import tempfile
import argparse
import srctools
import numpy as np
import srctools.bsp
import srctools.mdl
import srctools.game
import open3d as o3d
import srctools.filesys
 
from functools import lru_cache
from meshlib import mrmeshpy as mm
import SourceIO.library.models.vvd as vvd
import SourceIO.library.utils as vtxutils
import SourceIO.library.models.mdl.v49 as mdl
import SourceIO.library.models.vtx.v7.vtx as vtx

parser = argparse.ArgumentParser(__file__)

parser.add_argument('FILE')

parser.add_argument('--filetype', '-ft', help='what type is FILE', choices=['bsp', 'portal', 'bsp_visleaf'])
parser.add_argument('--proptype', '-ap', help='entity classname to also render, with optional keyvalue name, formatted as classname.keyvalue. repeatable.', action='append')
parser.add_argument('--game', '-g', type=str, help='path to folder with game\'s main gameinfo.txt. must be specified if filetype=bsp and enableprops=true.')
parser.add_argument('--enableprops', '-ep', help='set to false to disable props in BSP mode', choices=['true', 'false'], default='true')
parser.add_argument('--enablebrushes', '-eb', help='set to false to disable basic map geometry in BSP mode', choices=['true', 'false'], default='true')
parser.add_argument('--upwardaxis', '-up', help='model\'s up axis. anything but +Z will preview incorrectly in most software (STL usually has +Z as up) but programs expecting other axes will work fine', choices=['+Z', '+Y', '+X'], default='+Z')
#parser.add_argument('--crowbar', '-cr', type=str, help='path to CrowbarDecompiler.exe (https://github.com/mrglaster/Source-models-decompiler-cmd/releases/latest). required if props are enabled in BSP mode')

args = parser.parse_args()

portalregex = re.compile(r'\([-0-9\.]+ [-0-9\.]+ [-0-9\.]+ \)')
numregex = re.compile(r'[-0-9\.]+')
vertexregex = re.compile(r'[-0-9\.]+ ([-0-9\.]+) ([-0-9\.]+) ([-0-9\.]+) ([-0-9\.]+) ([-0-9\.]+) ([-0-9\.]+)')

def rotateModel(model: mm.Mesh, angles: srctools.math.Angle) -> mm.Mesh:
    '''
    Rotate model by `angles` degrees and return result.
    '''
    model.transform(mm.AffineXf3f.linear(mm.Matrix3f.rotation(mm.Vector3f.plusY(), math.radians(angles.pitch))))
    model.transform(mm.AffineXf3f.linear(mm.Matrix3f.rotation(mm.Vector3f.plusX(), math.radians(angles.roll))))
    model.transform(mm.AffineXf3f.linear(mm.Matrix3f.rotation(mm.Vector3f.plusZ(), math.radians(angles.yaw))))
    
    return model

rot_offsets = {
    '+Z': (0, 0, 0),
    '+Y': (0, 0, -90),
    '+X': (90, 0, 0),
}
final_mesh = mm.Mesh()

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

def read_vtx_and_build_mesh(mdl_filepath: str, vtx_filepath: str, vertices: list[tuple[float, float, float]]):
    # Thank you so much craftablescience!!
    
    vtxfile = vtx.Vtx.from_buffer(vtxutils.FileBuffer(vtx_filepath))
    mdlfile = mdl.MdlV49.from_buffer(vtxutils.FileBuffer(mdl_filepath))
    
    gindices = []
    
    def addIndex(mmesh, mmodel, stripgroup, index):
        gindices.append(int(stripgroup.vertexes[index]['original_mesh_vertex_index'][0])
                       + mmesh.vertex_index_start + mmodel.vertex_offset)
    
    for mbodypart, vbodypart in zip(mdlfile.body_parts, vtxfile.body_parts):
        for mmodel, vmodel in zip(mbodypart.models, vbodypart.models):
            for mmesh, vmesh in zip(mmodel.meshes, vmodel.model_lods[0].meshes):
                for vstripgroup in vmesh.strip_groups:
                    for vstrip in vstripgroup.strips:
                        indices = vstripgroup.indices[vstrip.index_mesh_offset : vstrip.index_mesh_offset + vstrip.index_count].tolist()
                        if vstrip.flags & 2 == 2:
                            # exclude the two last ones
                            for i, index in enumerate(indices[:-2]):
                                addIndex(mmesh, mmodel, vstripgroup, indices[i+0])
                                if i % 2 == 0:
                                    addIndex(mmesh, mmodel, vstripgroup, indices[i+2])
                                    addIndex(mmesh, mmodel, vstripgroup, indices[i+1])
                                else:
                                    addIndex(mmesh, mmodel, vstripgroup, indices[i+1])
                                    addIndex(mmesh, mmodel, vstripgroup, indices[i+2])
                        elif vstrip.flags & 1 == 1:
                            for i in range(0, len(indices), 3):
                                addIndex(mmesh, mmodel, vstripgroup, indices[i])
                                addIndex(mmesh, mmodel, vstripgroup, indices[i+2])
                                addIndex(mmesh, mmodel, vstripgroup, indices[i+1])
          
    # convert indices to triangles
    triangles = []
    it = iter(gindices)
    while (triangle := [next(it, None), next(it, None), next(it, None)]) and triangle != [None, None, None]:
        triangles.append(triangle)
          
    #print(list(gindices))                          
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()  # optional but helps visualization

    o3d.io.write_triangle_mesh('PropTest.stl', mesh)
    return mesh

def vvd_to_mesh_file(vertices: list[tuple[float, float, float]], mdl_filepath: str, vtx_filepath: str, lod: int=1, output_file: str="output.ply"):
    mesh = read_vtx_and_build_mesh(mdl_filepath, vtx_filepath, vertices)
    mesh.compute_vertex_normals()
    #o3d.visualization.draw_geometries([mesh])

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

def readVVD(d: str, file: srctools.mdl.FileRSeek) -> list[tuple[float, float, float]]:
    '''
    Read a VVD file from a Filesystem.
    
    :param file: File obtained from a Filesystem.
    :type file: srctools.mdl.FileRSeek
    :return: A list of 3-tuples for vertex coordinates.
    :rtype: list[tuple[float, float, float]]
    '''
    
    with open(os.path.join(d, 'vvd.vvd'), 'wb') as f:
        f.write(file.read())
    
    vvdfile = vvd.Vvd.from_buffer(vtxutils.FileBuffer(os.path.join(d, 'vvd.vvd')))
    vertices = []
    for lod in vvdfile.lod_data:
        for vertex in lod:
            vertices.append(vertex['vertex'].tolist())
            
    return vertices

def addProp(prop: srctools.bsp.StaticProp, fs: srctools.filesys.FileSystemChain):
    with tempfile.TemporaryDirectory('-v3mf') as d:
        m = mm.Mesh(makeMeshFromSourceModel(prop.model, d, fs))
        
        angles = prop.angles
        
        rotateModel(m, angles)
        m.transform(mm.AffineXf3f.translation(mm.Vector3f(*prop.origin)))
        final_mesh.addMesh(m)

@lru_cache
def makeMeshFromSourceModel(name: str, d: str, fs: srctools.filesys.FileSystemChain) -> dict[str, mm.Mesh | float]:
    #print('    extract files...')
    #for ext in srctools.mdl.MDL_EXTS:
    #    newname = name.replace('.mdl', ext)
    #    try:
    #        with fs.open_bin(newname) as f:
    #            with open(os.path.join(d, os.path.basename(newname)), 'wb') as f2:
    #                f2.write(f.read())
    #    except FileNotFoundError:
    #        pass
    #    
    ##print('    decompile...')
    #subprocess.run([args.crowbar, os.path.join(d, os.path.basename(name)), d], stdout=subprocess.PIPE, stderr=subprocess.PIPE).check_returncode()
    #
    ## now let's see if it saved as MODEL.smd or MODEL_reference.smd
    #if os.path.exists(os.path.join(d, os.path.basename(name.replace('.mdl', '.smd')))):
    #    fpath = os.path.join(d, os.path.basename(name.replace('.mdl', '.smd')))
    #elif os.path.exists(os.path.join(d, os.path.basename(name.replace('.mdl', '_reference.smd')))):
    #    fpath = os.path.join(d, os.path.basename(name.replace('.mdl', '_reference.smd')))
    #elif os.path.exists(os.path.join(d, os.path.basename(name.replace('.mdl', '_model.smd')))):
    #    fpath = os.path.join(d, os.path.basename(name.replace('.mdl', '_model.smd')))
    ## if there's still no good path, just find the first one that isn't an LOD or physics
    #else:
    #    for file in os.listdir(d):
    #        fp = os.path.join(d, file)
    #        # TODO: this could fail for models that have "physics" in the actual model name
    #        if os.path.isfile(fp) and 'lod' not in file and 'physics' not in file and file.endswith('.smd'):
    #            fpath = os.path.join(d, file)
    #            
    #    # if still no results        
    #    if fpath is None:
    #        print(os.listdir(d))
    #        raise FileNotFoundError('failed to find decompiled SMD')
#
    ##print('    convert to mesh...')
    #smd_to_mesh_file(fpath, os.path.join(d, 'modelmesh.stl'))
    #
    with open(os.path.join(d, 'vtx.vtx'), 'wb') as f:
        with fs.open_bin(name.replace('.mdl', '.dx90.vtx')) as f2:
            f.write(f2.read())
            
    with open(os.path.join(d, 'mdl.mdl'), 'wb') as f:
        with fs.open_bin(name) as f2:
            f.write(f2.read())
    
    vertices = readVVD(d, fs.open_bin(name.replace('.mdl', '.vvd')))
    vvd_to_mesh_file(vertices, os.path.join(d, 'mdl.mdl'), os.path.join(d, 'vtx.vtx'), 1, os.path.join(d, 'modelmesh.stl'))
    
    #print('    add mesh to output...')
    m = mm.loadMesh(os.path.join(d, 'modelmesh.stl'))
    
    return m
match args.filetype:
    case "bsp":
        bsp = srctools.bsp.BSP(args.FILE, srctools.bsp.VERSIONS.PORTAL_2)
        if args.enablebrushes == 'true':
            print("constructing base map model")
            for entity, bmodel in bsp.bmodels.items():
                #print(entity['targetname'])
                for face in bmodel.faces:
                    cl = mm.PointCloud()
                    for edge in face.edges:
                        #cl.addPoint(mm.Vector3f(*edge.a))
                        cl.addPoint(mm.Vector3f(*edge.b))
                    mesh = mm.triangulatePointCloud(cl)
                    final_mesh.addMesh(mesh)
                break
            
        if args.enableprops != 'false':
            if getattr(args, 'game', None) is None:
                raise Exception('must specify --game if props are enabled')
            #if getattr(args, 'crowbar', None) is None:
            #    raise Exception('must specify --crowbar if props are enabled')
            
            print("creating filesystem chain")
            game = srctools.game.Game(args.game)
            
            fs = game.get_filesystem()
            # mount BSP
            fs.add_sys(srctools.filesys.ZipFileSystem(args.FILE, zipfile=bsp.pakfile))
            
            print("populating static props")
            bar = tqdm.tqdm(bsp.props, unit=' props')
            for prop in bar:
                addProp(prop, fs)
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
                    
                nefertiti_mesh = mm.triangulatePointCloud(cl)
                final_mesh.addMesh(nefertiti_mesh)
                
    case "bsp_visleaf":
        bsp = srctools.bsp.BSP(args.FILE, srctools.bsp.VERSIONS.PORTAL_2)
        
        tree = bsp.vis_tree()
        for leaf in tree.iter_leafs():
            for face in leaf.faces:
                cl = mm.PointCloud()
                for edge in face.edges:
                    #cl.addPoint(mm.Vector3f(*edge.a))
                    cl.addPoint(mm.Vector3f(*edge.b))
                mesh = mm.triangulatePointCloud(cl)
                final_mesh.addMesh(mesh)
                
rotateModel(final_mesh, srctools.math.Angle(*rot_offsets[args.upwardaxis]))
mm.saveMesh(final_mesh, "Mesh.stl")