# paperiano

Play a piano drawn on paper using Computer Vision and Deep Learning.

## Installation

### Preparation of the image dataset

1. Download videos, and place them under a `./data/videos` directory.
2. Extract frames from the videos:

```bash
python cli.py readframes data/videos/flying.MOV data/images --start 15 --end 145
python cli.py readframes data/videos/touching.MOV data/images --start 0 --end 120
```
