import cv2
import random
import numpy as np
from PIL import Image


def VideoDecoder(results):
    file_path = results['filename']
    results['format'] = 'video'

    cap = cv2.VideoCapture(file_path)
    videolen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sampledFrames = []
    for i in range(videolen):
        ret, frame = cap.read()
        # maybe first frame is empty
        if ret == False:
            continue
        img = frame[:, :, ::-1]
        sampledFrames.append(img)
    results['frames'] = sampledFrames
    results['frames_len'] = len(sampledFrames)
    return results


def Sampler(results, num_seg=8, seg_len=1, valid_mode=True):
    frames_len = int(results['frames_len'])
    average_dur = int(frames_len / num_seg)
    frames_idx = []

    for i in range(num_seg):
        idx = 0
        if not valid_mode:
            if average_dur >= seg_len:
                # !!!!!!!
                idx = random.randint(0, average_dur - seg_len)
                # idx = 0
                idx += i * average_dur
            elif average_dur >= 1:
                idx += i * average_dur
            else:
                idx = i
        else:
            if average_dur >= seg_len:
                idx = (average_dur - 1) // 2
                idx += i * average_dur
            elif average_dur >= 1:
                idx += i * average_dur
            else:
                idx = i
        for jj in range(idx, idx + seg_len):
            if results['format'] == 'video':
                frames_idx.append(int(jj % frames_len))
            else:
                raise NotImplementedError

    return Processing_Sampler(frames_idx, results)


def Processing_Sampler(frames_idx, results):
    data_format = results['format']
    if data_format == 'video':
        frames = np.array(results['frames'])
        imgs = []
        for idx in frames_idx:
            imgbuf = frames[idx]
            img = Image.fromarray(imgbuf, mode='RGB')
            imgs.append(img)
        results['imgs'] = imgs
    return results


def Scale(results, short_size=256, fixed_ratio=False, do_round=False):
    imgs = results['imgs']
    resized_imgs = []
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.size
        if (w <= h and w == short_size) or (h <= w and h == short_size):
            resized_imgs.append(img)
            continue
        if w < h:
            ow = short_size
            if fixed_ratio:
                oh = int(short_size * 4.0 / 3.0)
            else:
                oh = int(round(h * short_size /
                               w)) if do_round else int(
                    h * short_size / w)
        else:
            oh = short_size
            if fixed_ratio:
                ow = int(short_size * 4.0 / 3.0)
            else:
                ow = int(round(w * short_size /
                               h)) if do_round else int(
                    w * short_size / h)
        resized_imgs.append(img.resize((ow, oh), Image.BILINEAR))
    results['imgs'] = resized_imgs
    return results


def CenterCrop(target_size, results, do_round=True):
    imgs = results['imgs']
    ccrop_imgs = []
    for img in imgs:
        w, h = img.size
        th, tw = target_size, target_size
        assert (w >= target_size) and (h >= target_size), \
            "image width({}) and height({}) should be larger than crop size".format(
                w, h, target_size)
        x1 = int(round((w - tw) / 2.0)) if do_round else (w - tw) // 2
        y1 = int(round((h - th) / 2.0)) if do_round else (h - th) // 2
        ccrop_imgs.append(img.crop((x1, y1, x1 + tw, y1 + th)))
    results['imgs'] = ccrop_imgs
    return results


def Image2Array(results):
    imgs = results['imgs']
    np_imgs = (np.stack(imgs)).astype('float32')
    transpose = True
    if transpose:
        np_imgs = np_imgs.transpose(0, 3, 1, 2)  # tchw
    results['imgs'] = np_imgs
    return results


def Normalization(results, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], tensor_shape=[3, 1, 1]):
    imgs = results['imgs']
    mean = np.array(mean).reshape(tensor_shape).astype(np.float32)
    std = np.array(std).reshape(tensor_shape).astype(np.float32)
    norm_imgs = imgs / 255.0
    norm_imgs -= mean
    norm_imgs /= std
    results['imgs'] = norm_imgs
    return results


def preprocessing(test_file_path):
    results = dict()
    results['filename'] = test_file_path
    results = VideoDecoder(results)
    results = Sampler(results)
    results = Scale(results)
    results = Image2Array(results)
    results = Normalization(results)
    data = results['imgs']
    data = np.reshape(data, (1, data.shape[0], data.shape[1], data.shape[2], data.shape[3]))
    return data