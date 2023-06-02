<!-- WARNING - this repo if sort of... raw. There's some hacks done to 
accomodate for the train.py script (hint: see all new "# TODO:" tags in main.py, train.py, example-config.yaml) -->
# Iterative Embedding Refinement - Generating Flexible Text Embeddings of Animated Character Illustrations
Pipeline             |  Image Filtering
:-------------------------:|:-------------------------:
![image](https://github.com/Shr1ftyy/flexible_embeddings/assets/49330057/b84f42c8-f855-4dee-9ed6-3b17ac8b72f8) | ![image](https://github.com/Shr1ftyy/flexible_embeddings/assets/49330057/7d8564a9-44b9-49ec-a4ad-8840ce93cb8c)


This repo borrows heavily from https://github.com/rinongal/textual_inversion/tree/424192de1518a358f1b648e0e781fdbe3f40c210


1. Follow the setup instructions from the textual inversion repo (link above ) like installing sd 1.4

3. Open `./example-config.yaml`, read it's instructions and edit it 

4. Run `train.py`, e.g.:
```
python train.py --config example-config.yaml
```
Sometimes, you might get the following error:
```
Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1 library.
        Try to import numpy first or set the threading layer accordingly. Set MKL_SERVICE_FORCE_INTEL to force it.
```
in which case, run the command with the env variable `MKL_SERVICE_FORCE_INTEL` set to `1` :
```
MKL_SERVICE_FORCE_INTEL=1 python train.py --config example-config.yaml
``` 

All embeddings are stored under folders following the following scheme:
```
logs/_{embedding_name}-{iteration}/checkpoints
```
For example, the final checkpoint of the 3rd iteration of the "bluhrboi" embedding would be at:
 ```
logs/_bluhrboi-2/checkpoints/embeddings.pt
```
