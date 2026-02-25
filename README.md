# v3mf - Source to STL converter
This tool can convert a compiled BSP map to an STL 3D model for printing out memento figurines of your maps! Note that it may require some post-processing as nodraw faces do not generate triangles.

## Features
Modes:
- Portals - create planes at portal intersections (may be removed in the future)
- BSP (geometry) - brushes and static props. Model files are automatically found within one of the paths defined in your gameinfo.txt or the BSP itself
- BSP (visleafs) - replicates shapes of rooms (must compile with VVIS)

### Future features
- Ability to specify custom entity classes to add as props (parameter added but not implemented yet)
- Specify export filename
- Autodetect import file type
- GUI?

## Name
"v3mf" is a combination of "VMF", which is Valve's map file format (unfortunately unsupported by v3mf due to how complex brush notation in it is), and "3MF", which is a file format often used for additive manufacturing (i.e. 3D printing).

## Special thanks
- craftablescience (tysm craftable!!!)
- creators and contributors of SourceIO

## Building from source
Windows: `build.bat`
- If your user has non-ASCII characters in its path (e.g. cyrillics), create a new user with a purely-ASCII name and use `runas /user:<YOUR ALTERNATE USER'S NAME> /env build.bat`. *This is a bug in Scons, which is used by the build system.*
Linux `sh build.sh`
- I don't know if the non-ASCII username bug also applies for Linux, but, if it does, use `su -c 'sh build.sh' <YOUR ALTERNATE USE USER'S NAME>"` or `sudo -u <YOUR ALTERNATE USER'S NAME> 'sh build.sh'`.

You might also have issues if the path to the code has unicode characters in it. In that case, move it someplace else or use a symlink to the folder and enter the symlink.
