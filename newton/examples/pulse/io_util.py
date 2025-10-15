import numpy as np
import OpenGL.GL
from PIL import Image


def dump_gl_frame_image(width, height, filename="imgui_capture.png"):
    # Read pixels from the framebuffer (bottom-to-top order)
    data = OpenGL.GL.glReadPixels(0, 0, width, height, OpenGL.GL.GL_RGBA, OpenGL.GL.GL_UNSIGNED_BYTE)

    # Convert to NumPy array
    image = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 4))

    # Flip vertically (OpenGL’s origin is bottom-left)
    image = np.flipud(image)

    # Save as image using PIL
    Image.fromarray(image).save(filename)