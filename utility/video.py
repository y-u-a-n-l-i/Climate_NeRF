import os 
import cv2 
try:
    # for backward compatibility
    import imageio.v2 as imageio
except ModuleNotFoundError:
    import imageio
from PIL import Image
import numpy as np
import argparse

def extract_frames():
    parser = argparse.ArgumentParser()
    parser.add_argument('-video', help='path to video')
    parser.add_argument('-outdir', help='output folder')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    vidcap = cv2.VideoCapture(args.video)
    i = 0
    while True:
        success, image = vidcap.read()
        if not success:
            break
        path = os.path.join(args.outdir, '{:0>5d}.png'.format(i))
        cv2.imwrite(path, image)
        i += 1

def read_video_frames(video_path):
    vidcap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        success, image = vidcap.read()
        if not success:
            break
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(image)
    return frames

def is_image_name(name):
    valid_names = ['.jpg', '.png', '.JPG', '.PNG']
    for v in valid_names:
        if name.endswith(v):
            return True
    return False

def generate_video():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir')
    parser.add_argument('-out')
    args = parser.parse_args()

    imgs = sorted([os.path.join(args.dir, img) for img in os.listdir(args.dir) if is_image_name(img)])
    imgs = [np.array(Image.open(img)) for img in imgs]

    imageio.mimsave(args.out, imgs, fps=30, macro_block_size=1)

    # imgs = [Image.open(img) for img in imgs]
    # imgs += imgs[::-1]
    # imgs[0].save(args.out, format='GIF', append_images=imgs,
    #      save_all=True, duration=10, loop=0)


def merge_video():
    parser = argparse.ArgumentParser()
    parser.add_argument('-left')
    parser.add_argument('-right')
    parser.add_argument('-out')
    args = parser.parse_args()

    frames_left  = read_video_frames(args.left)
    frames_right = read_video_frames(args.right)
    frames_out   = []

    frame_len = min(len(frames_left), len(frames_right))
    for i in range(frame_len):
        left  = frames_left[i]
        right = frames_right[i]
        out = np.concatenate([left, right], axis=1)
        frames_out.append(out)

    imageio.mimsave(args.out,
                    frames_out,
                    fps=30, macro_block_size=1)

if __name__ == '__main__':
    # extract_frames()
    # generate_video()
    merge_video()