# Multi-Image Generation with In-Context LoRA

This repo aims to generate coherent 2D multi-view scenes (multiple images with intrinsic relationships, such as different viewpoints of the same object or scene) using minimal training data and limited compute. The approach builds on In-Context LoRA, a method to adapt diffusion transformer (DiT) models to multi-image outputs without changing the model architecture.

The key idea is to concatenate multiple images into one larger image during training, use a joint caption describing all sub-images, and fine-tune only lightweight LoRA adapters on a small dataset (on the order of 10–100 image sets). This enables high-fidelity multi-image generation that adheres well to the prompt, even with limited data.

# 1. Dataset Preparation (Multi-View Image Sets)
Collect or curate a small set of multi-view image groups. Each group should contain a few images that are related – e.g. different views of the same object or scene, or a sequence of images with a consistent theme or identity. You can use an existing multiview dataset like MVImgNet as a source: MVImgNet contains multi-view images of ~220k real-world objects across 238 classes.The goal  for an MVP is a minimal viable dataset so about 10–20 image sets are sufficient (each set might have e.g. 2–4 images of a given object/scene from various angles or contexts).

# 2.Automatic Caption Generation for Multi-Image Scenes

For each image set, we need a single descriptive caption that encompasses all views/images. Writing these by hand is possible but to ensure scalability and consistency, we can automate caption generation using multimodal models:

Image captioning models (BLIP-2, InstructBLIP, etc.) – You can use a model like BLIP-2 to caption each image individually. BLIP-2 can generate a sentence describing an image’s content. After captioning each image in the set, you will merge these descriptions into one multi-part prompt. 

Example (two-view caption): “[TWO-VIEWS] This set of two images presents a scene from two different viewpoints. [IMAGE1] The first image shows a living room with a sofa, side tables, a television, houseplants, wall decor, and a rug. [IMAGE2] The second image shows the same room from another angle, revealing additional details from the other side.”

Use the same markers in all your captions (e.g. always [IMAGE1], [IMAGE2], etc., or positional tags like [LEFT], [RIGHT] for two-image pairs
, [TOP], [BOTTOM] for two-image vertical pairs, etc. This consistency helps the model learn the structure of multi-image prompts. ). These tokens don’t carry inherent meaning, but during training the model will learn to associate them with positioning of sub-images.