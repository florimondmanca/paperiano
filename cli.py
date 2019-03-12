import os
from itertools import count
from pathlib import Path

import click
import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


class Video:
    def __init__(self, path):
        self.path = Path(path).absolute()

    def clip(self, start: int, end: int):
        resized_dir = str(self.path.parent / "resized")
        os.makedirs(resized_dir, exist_ok=True)
        dest = Path(resized_dir) / self.path.name
        ffmpeg_extract_subclip(self.path, start, end, targetname=dest)
        return Video(dest)

    def export_frames(self, out):
        out = Path(out)
        vidcap = cv2.VideoCapture(str(self.path))

        for i in count(0):
            success, image = vidcap.read()
            if not success:
                break
            path = out / f"{self.path.stem}_{i}.jpg"
            cv2.imwrite(str(path), image)
            print("Frame", i)


@click.group()
def cli():
    pass


@cli.command()
@click.argument("file_in", type=click.Path(exists=True, dir_okay=False))
@click.argument("dir_out", type=click.Path(exists=True))
@click.option("--start", type=int)
@click.option("--end", type=int)
def readframes(file_in, dir_out, start, end):
    video = Video(file_in)

    if start is not None and end is not None:
        video = video.clip(start, end)

    video.export_frames(dir_out)


if __name__ == "__main__":
    cli()
