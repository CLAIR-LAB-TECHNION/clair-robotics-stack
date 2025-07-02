import os
print("DISPLAY =", os.environ.get("DISPLAY"))

import glfw

if not glfw.init():
    raise Exception("GLFW init failed")