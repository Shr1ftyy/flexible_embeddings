WARNING - this repo if sort of... raw. There's some hacks done to 
accomodate for the train.py script (hint: see all new "# TODO:" tags in main.py, train.py, config.yaml)

This repo borrows heavily from https://github.com/rinongal/textual_inversion/tree/424192de1518a358f1b648e0e781fdbe3f40c210


1. follow the setup instructions from the textual inversion repo (link above )
1. Edit  `./config.yaml`, `configs/single_image.yaml`
2. run `train.py`, e.g.:
```
python train.py --config config.yaml
```