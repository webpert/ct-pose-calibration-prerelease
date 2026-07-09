rm -rf "./pose_gaussian/submodules/self-xray-gaussian-rasterization-voxelization/build"
pip uninstall -y self_xray_gaussian_rasterization_voxelization
pip install -e ./pose_gaussian/submodules/self-xray-gaussian-rasterization-voxelization --no-build-isolation