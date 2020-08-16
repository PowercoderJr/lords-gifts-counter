# Lords Mobile gifts counter

Capture your gifts list on video, feed it to **analyze.py** and get structured info about from whom, how many and of what rarity gifts you got. Use `-dm 1` argument for better understanding what you should provide. You'll see 3 windows: _video_ - actually video but downscaled for convinience, _gift_roi_ - region where scrolled nicknames supposed to be, _sender_fragment_ - last detected sender nickname.

This script requires Tesseract OCR installed on your PC ([Windows](https://github.com/tesseract-ocr/tesseract/wiki#windows), [Linux](https://github.com/tesseract-ocr/tesseract/wiki), [Mac](https://github.com/tesseract-ocr/tesseract/wiki#macos)). Also don't forget to add it to PATH variable.

Based on in-game UI: russian, 2340x1080, 15.08.2020.

## Usage

    usage: python analyze.py [-h] [-s SCALE] [-of OUTPUT_FILENAME] [-dm {0,1,2}] video_filename

    Guild gifts counter for mobile game Lords Mobile for automatic accounting.

    positional arguments:
      video_filename        input video

    optional arguments:
      -h, --help            show this help message and exit
      -s SCALE, --scale SCALE
                            scale factor
      -of OUTPUT_FILENAME, --output_filename OUTPUT_FILENAME
                            output filename in *.xlsx format
      -dm {0,1,2}, --demonstration_mode {0,1,2}
                            (0) - do not display video and roi,
                            (1) - display video and roi at normal speed (Esc to stop),
                            (2) - display video and roi frame by frame (any key to move on, Esc to stop)

## Illustrations

### Demo

![demo](https://user-images.githubusercontent.com/25909264/90340978-6cea3200-e004-11ea-940f-040d48cac0f1.gif)

### Excel output

![excel_output](https://user-images.githubusercontent.com/25909264/90341004-999e4980-e004-11ea-999b-acf31f3a81ac.png)
