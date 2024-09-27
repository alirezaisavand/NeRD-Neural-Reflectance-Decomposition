import json
import os

import imageio
import numpy as np
import tensorflow as tf

import imageio

def trans_t(t):
    return tf.convert_to_tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1],], dtype=tf.float32,
    )


def rot_phi(phi):
    return tf.convert_to_tensor(
        [
            [1, 0, 0, 0],
            [0, tf.cos(phi), -tf.sin(phi), 0],
            [0, tf.sin(phi), tf.cos(phi), 0],
            [0, 0, 0, 1],
        ],
        dtype=tf.float32,
    )


def rot_theta(th):
    return tf.convert_to_tensor(
        [
            [tf.cos(th), 0, -tf.sin(th), 0],
            [0, 1, 0, 0],
            [tf.sin(th), 0, tf.cos(th), 0],
            [0, 0, 0, 1],
        ],
        dtype=tf.float32,
    )


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w

def get_image_from_exr(filename):
    import OpenEXR
    import Imath
    import numpy as np

    # Open the EXR file
    exr_file = OpenEXR.InputFile(filename)

    channels = exr_file.header()['channels'].keys()
    # Print the available channels
    # print("Available channels:", channels)

    # Get image size
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Read the image channels (R, G, B)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    red = np.frombuffer(exr_file.channel('Image.R', FLOAT), dtype=np.float32).reshape(height, width)
    green = np.frombuffer(exr_file.channel('Image.G', FLOAT), dtype=np.float32).reshape(height, width)
    blue = np.frombuffer(exr_file.channel('Image.B', FLOAT), dtype=np.float32).reshape(height, width)
    mask = np.frombuffer(exr_file.channel('Image.A', FLOAT), dtype=np.float32).reshape(height, width)
    # Combine channels into an RGB image
    rgb_image = np.stack([red, green, blue, mask], axis=-1)
    return rgb_image

def imread(f):
    if f.endswith("png"):
        return imageio.imread(f, apply_gamma=False)
    else:
        return imageio.imread(f)

def load_blender_data(basedir, half_res=False, trainskip=1, testskip=1, valskip=1):
    masksdir = os.path.join(basedir, 'masks')
    imagedir = os.path.join(basedir, 'images')
    splits = ["train", "val", "test"]
    metas = {}
    for s in splits:
        if s == 'val':
            tmp = 'train'
        else:
            tmp = s
        with open(os.path.join(basedir, "transforms_{}.json".format(tmp)), "r") as fp:
            metas[s] = json.load(fp)

    metas['val']['frames'] = [metas['train']['frames'][10 * i] for i in range(0, len(metas['train']['frames'])//10)]

    print('val size:', len(metas['val']['frames']))

    all_imgs = []
    all_masks = []
    all_poses = []
    all_ev100 = []
    counts = [0]
    meta = None
    for s in splits:
        meta = metas[s]
        imgs = []
        masks = []
        poses = []
        if s == "train":
            skip = max(trainskip, 1)
        elif s == "val":
            skip = max(valskip, 1)
        else:
            skip = max(testskip, 1)

        for frame in meta["frames"][::skip]:
            # fname = os.path.join(basedir, frame["file_path"])
            # img_file = (imageio.imread(fname) / 255).astype(np.float32)
            # img_file = get_image_from_exr(fname)
            image_name = 'Image' + frame["file_path"].split('\\')[1]
            image_path = os.path.join(imagedir, image_name)
            print('image path:', image_path)
            imgs.append(imread(image_path)[..., 0:3] / 255.0)

            mask_name = 'Segmentation' + frame["file_path"].split('\\')[1]
            mask_path = os.path.join(masksdir, mask_name)
            print('mask path:', mask_path)
            masks.append(np.expand_dims(imread(mask_path) / 255.0, axis=-1))

            # Read the poses
            poses.append(np.array(frame["transform_matrix"]))

            all_ev100.append(8)

        imgs = np.array(imgs).astype(np.float32)
        # Continue with the masks.
        # They only require values to be between 0 and 1
        # Clip to be sure
        masks = np.clip(np.array(masks).astype(np.float32), 0, 1)

        poses = np.array(poses).astype(np.float32)

        counts.append(counts[-1] + imgs.shape[0])

        print('imgs shape:', np.array(imgs).shape, 'masks shape:', np.array(masks).shape, 'poses shape', np.array(poses).shape)

        all_imgs.append(imgs)
        all_masks.append(masks)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0).astype(np.float32)
    masks = np.concatenate(all_masks, 0).astype(np.float32)
    poses = np.concatenate(all_poses, 0)
    ev100s = np.stack(all_ev100, 0).astype(np.float32)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    render_poses = tf.stack(
        [
            pose_spherical(angle, -30.0, 4.0)
            for angle in np.linspace(-180, 180, 40 + 1)[:-1]
        ],
        0,
    )

    if half_res:
        H = H // 2
        W = W // 2
        imgs = tf.image.resize(imgs, [H, W], method="area").numpy()
        masks = tf.image.resize(masks, [H, W], method="area").numpy()
        focal = focal / 2.0

    return imgs, masks, poses, ev100s, render_poses, [H, W, focal], i_split
