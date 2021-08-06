import PIL
import numpy as np


# Load image from file, convert to grayscale float32 numpy array
def load_image_grayscale(img_file):
    img = PIL.Image.open(img_file)

    # Convert to grayscale and return
    return img.convert("L")


def grayscale_resized_image(img):
    # Resize if image larger than 2k pixels on a side
    if img.size[0] > 2000 or img.size[1] > 2000:
        # print(f"Image too big ({img.size[0]} x {img.size[1]})")
        new_size = 800.0
        if img.size[0] > img.size[1]:
            # resize by width to new limit
            ratio = new_size / img.size[0]
        else:
            # resize by height
            ratio = new_size / img.size[1]
        # print("Reducing by factor of %.2g" % (1. / ratio))
        nx, ny = int(img.size[0] * ratio), int(img.size[1] * ratio)

        img = img.resize((nx, ny), PIL.Image.ADAPTIVE)
        # print(f"New size: ({img.size[0]}px x {img.size[1]}px)")

    # Convert to grayscale and array
    return np.asarray(img.convert('L'), dtype=np.float32)


def grayscale_image(img):
    # Convert to grayscale and array
    return np.asarray(img.convert('L'), dtype=np.float32)