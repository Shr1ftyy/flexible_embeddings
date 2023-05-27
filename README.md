WARNING - this repo if sort of... raw. There's some hacks done to 
accomodate for the train.py script (hint: see all new "# TODO:" tags in main.py, train.py, config.yaml)

This repo borrows heavily from https://github.com/rinongal/textual_inversion/tree/424192de1518a358f1b648e0e781fdbe3f40c210


1. follow the setup instructions from the textual inversion repo (link above )
1. Edit  `./config.yaml`, `configs/single_image.yaml`
2. run `train.py`, e.g.:
```
python train.py --config config.yaml
```

sometimes, you might get the following error:
```
Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library.
        Try to import numpy first or set the threading layer accordingly. Set MKL_SERVICE_FORCE_INTEL to force it.
```

in which case, just do this:

```
MKL_SERVICE_FORCE_INTEL=1 python train.py --config config.yaml
```