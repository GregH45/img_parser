# Image parse
Small code to check avatars beeing uploaded.

## Criterias: 
- Picture must be at least 512x512 px
- Picture must be circular (Transparent background allowed)
- The picture has to qualify for beeing "Happy" (determined via red yellow and orange color presence)

## How to use: 
Download the source code and install requirements: 

```bash 
git clone git@github.com:GregH45/img_parser.git
pip install -r requirements.txt
python3 main.py path_to_images
```

## Dependencies 
Uses opencv-python and numpy