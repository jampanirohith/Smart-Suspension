from roboflow import Roboflow
rf = Roboflow(api_key="xlS3pb3QgzjFDpIbMZlJ")
project = rf.workspace("jampanis-workspace").project("pothole-small-lmiwu")
version = project.version(1)
dataset = version.download("yolov8")
                