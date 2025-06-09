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

After generation, review and refine the captions. Ensure they correctly describe each image and emphasize the common elements between views (so the model learns what should stay consistent) as well as the differences (so the model knows each panel can vary appropriately). The captions should read like a single narrative or list of observations rather than disconnected sentences.

# 3. Preprocessing: Composite Images and Merged Prompts

Turn each multi-image set into the paired training data for the model.
- **Concatenate Images**: Concatenate the images in each set into a single larger image. For example, for two images, you can place them side by side or one above the other, for four images, a 2×2 grid is convenient. Ensure the composite image has a consistent size and aspect ratio across your dataset. The idea is to mimic how the model will output multiple images in one go. Arrange images in a consistent order and orientation (the order should match the order in your caption). Add minimal spacing or dividing lines if needed (but typically just concatenating directly is fine so the model sees one continuous image).

 - **Composite image dimensions**: Choose a fixed size for composite images, e.g. 512x1024 for two images side by side, or 1024x1024 for four images in a grid. This ensures uniformity and helps the model learn to generate multi-image outputs. You can also stack vertically or in grid; just ensure your caption format corresponds (e.g., if you do a vertical stack of three, maybe use [TOP], [MIDDLE], [BOTTOM] markers or [IMAGE1]/[IMAGE2]/[IMAGE3] in top-to-bottom order). Keep in mind the final composite size affects memory

 - **Prompt concatenation**: Concatenate the individual image captions into a single loooong prompt for the model. This prompt should maintain the same structure and markers as the individual captions, ensuring the model understands the relationships between the images. We now have training pairs of the form: *(composite_image, multi_image_caption)*.

- **Data Structure**: Your training data should be structured as a list of tuples, where each tuple contains the composite image and its corresponding multi-image caption.If using diffusers or similar training scripts, you might save each composite image as a file and put the caption text in a .txt file with the same name.
    - Example data structure: *train_data/scene01.jpg – and a file train_data/scene01.txt containing the caption.*

# 4. Fine-tuning In-Context LoRA on the FLUX model
To train the model, we will use the In-Context LoRA approach on a diffusion transformer model like FLUX.

LoRA (Low-Rank Adaptation) inserts trainable low-rank weight matrices into the model (typically into attention layers) and freezes the original model weights. This drastically reduces the number of parameters that need updating (and thus memory usage), making it feasible to train on a single high-end GPU.