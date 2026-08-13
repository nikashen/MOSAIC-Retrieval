# Third-party notices and publication boundary

The MIT license in this repository applies only to original project source and documentation. It does not grant rights to datasets, images, videos, captions, pretrained model weights, or third-party packages.

## COCO 2017

COCO assets are not included in this repository. COCO annotations and the COCO website are distributed under CC BY 4.0; individual images remain governed by their original Flickr licenses. Users must follow the COCO terms and each image's license. No COCO image, caption-bearing screenshot, or annotation row is included in this public snapshot.

## OpenAI CLIP / Hugging Face Transformers

The declared checkpoint is `openai/clip-vit-base-patch32`. It is loaded at runtime by `transformers`; this repository does not redistribute its model weights. Follow the checkpoint model card and license terms.

## MSR-VTT local evaluation mirror

The historical video experiment used the revision-pinned `friedrichor/MSR-VTT` mirror. That mirror does not declare a dataset license. Videos, captions, derived frame features, query-level rankings and checkpoints are excluded from Git, wheels, containers, Release assets and Pages. The public package does not automatically download the mirror: users must provide a locally authorized dataset copy and independently establish permission for any use.

The Pages site contains aggregate metrics and code-drawn synthetic cards only. They are interface examples, not experimental evidence or reconstructed rankings.

## Python and system dependencies

PyTorch, Transformers, Pillow, NumPy, SciPy, scikit-learn, FAISS, FastAPI, Uvicorn, safetensors and imageio-ffmpeg remain under their respective upstream licenses. This repository does not redistribute their wheels or binaries. The Dockerfile is a build recipe, not a prebuilt image release; review transitive notices, including the FFmpeg binary bundled by imageio-ffmpeg, before distributing an image.
