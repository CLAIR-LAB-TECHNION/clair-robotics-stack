import pyrealsense2 as rs
import json

def intrinsics_to_dict(intr):
    return {
        "width": intr.width,
        "height": intr.height,
        "ppx": intr.ppx,
        "ppy": intr.ppy,
        "fx": intr.fx,
        "fy": intr.fy,
        "model": str(intr.model),  # Convert the model enum to string
        "coeffs": list(intr.coeffs)  # Convert the coeffs to a list
    }

# Configure the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(config)

# Get the intrinsics of the depth and color streams
depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

# Get the extrinsics between depth and color streams
depth_to_color_extrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to(profile.get_stream(rs.stream.color))
color_to_depth_extrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_extrinsics_to(profile.get_stream(rs.stream.depth))

# Save the parameters to a JSON file
params = {
    "depth_intrinsics": intrinsics_to_dict(depth_intrinsics),
    "color_intrinsics": intrinsics_to_dict(color_intrinsics),
    "depth_to_color_extrinsics": {
        "rotation": list(depth_to_color_extrinsics.rotation),
        "translation": list(depth_to_color_extrinsics.translation)
    },
    "color_to_depth_extrinsics": {
        "rotation": list(color_to_depth_extrinsics.rotation),
        "translation": list(color_to_depth_extrinsics.translation)
    }
}

with open("camera_params.json", "w") as f:
    json.dump(params, f, indent=4)

pipeline.stop()
