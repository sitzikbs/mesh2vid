Introduction
============
This is a repo to simply and easily convert 3D mesh to 2D video. 

# Requirements
- Python 3.10
- trimsh
- pyrender
- opencv
- numpy



# Installation
````
conda create -n mesh2vid python=3.10
conda activate mesh2vid
````

# install dependencies
```
conda install trimsh -c conda-forge -y
conda install pyrender -c conda-forge -y
conda install opencv -c conda-forge -y

```

# Usage
````
python mesh2vid.py --input_mesh_path ./meshes/3d_mesh.obj --output_video_path ./output/3d_mesh_vid.mp4
````