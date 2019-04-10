import cv2
import matplotlib.pyplot as plt
import numpy as np

from typing import Any, Tuple

Hist = Any
Image = np.ndarray


def load(path: str, gray: bool = True) -> Tuple[Image, tuple]:
    if gray:
        GRAYSCALE = 0
        img = cv2.imread(path, GRAYSCALE)
    else:
        COLOR = 1
        bgr = cv2.imread(path, COLOR)
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return img, img.shape


def show(img: Image, vmin: int = 0, vmax: int = 255, gray: bool = True):
    kwargs: dict = {}
    if gray:
        kwargs.update(cmap="gray", vmin=vmin, vmax=vmax)
    plt.figure(figsize=(10, 7))
    plt.imshow(img, interpolation="nearest", **kwargs)


def to_gray(img: Image):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def histogram(img: Image) -> Hist:
    channels = [0]  # only one channel for grayscale
    mask = None  # use the full image
    bins = [256]  # use 256 bins, i.e. one per gray level
    ranges = [0, 256]  # values range from 0 to 256 (not included).
    return cv2.calcHist([img], channels, mask, bins, ranges)


def hist_plot(hist: Hist):
    plt.plot(hist)
    plt.xlabel("Level")
    plt.ylabel("Intensity")
