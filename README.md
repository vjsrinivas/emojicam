# EmojiCam

<img src='thumbnail.gif' style="display: block; margin-left: auto; margin-right: auto; width: 50%;">
<br>

**Description:** Translate images to emoticons 👏😌

## Requirements:
- OpenCV2
- NumPy
- Scipy
- Sklearn
- DepthAI (optional - only if you have a DepthAI camera)

## Trying it yourself:

1. Clone this GitHub project: `git clone `
2. Install requirements: `pip install -r requirements.txt`
3. Download Twitter Emoji library [here](https://github.com/twitter/twemoji/archive/refs/heads/master.zip), extract the `72x72` folder under `./assets/`, and rename folder to "twitter".
4. Run script:
    * Calculating from scratch: `python main.py` (**NOTE:** Calculating cache will take ~1.5 hours )
    * Calculating from cache is just the same command. You can get the [cache here](https://drive.google.com/file/d/1I6Y5ihons7G3wFlIW5tosuAgPTbC5GtT/view?usp=sharing)
    
**Performance:** On my machine (AMD Ryzen 2700X), I was able to maintain 38-42 FPS. If using a lower-spec CPU (this is pretty CPU dependent), then try:
   - Decreasing video size
   - Increasing pixelization size

## Main Usage

```
usage: main.py [-h] [--video VIDEO] [--output OUTPUT] [--depthai]

optional arguments:
  -h, --help       show this help message and exit
  --video VIDEO    Video input. Enter either a number for a webcam device or
                   path to a video file. Example: python main.py --video
                   test.mp4
  --output OUTPUT  Video output path. Enter a string to the path where the
                   video output will be saved. Default is a .mp4 file.
                   Example: python main.py --output out.mp4
  --depthai        Optional flag only used for DepthAI devices. Flag is false
                   by default. After enabling program will import depthai
                   module.
```

## Future Work:

Optimize algorithm, rethink mapping algorithm, and program as some kind of filter for a social media platform.
Hmmmm.... could multithreading help speed things up?
