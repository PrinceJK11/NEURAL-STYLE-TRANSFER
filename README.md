# NEURAL-STYLE-TRANSFER

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: SOURAV PAL CHAUDHURI

*ITERN ID*: CT08DL242

*DOMAIN*: AI

*DURATION*: 8 WEEKS

*MENTOR*: NEELA SANTOSH

# 🎨 Neural Style Transfer with PyTorch

This project performs **Neural Style Transfer** using a pre-trained VGG-19 network from PyTorch’s `torchvision.models`. The goal is to blend the *content* of one image with the *style* of another to create a visually artistic result.

## 📸 Example

<p align="center">
  <img src="stylized_output.jpg" alt="Stylized Output" width="500"/>
</p>

## 🧠 What is Neural Style Transfer?

Neural Style Transfer is a technique that uses deep neural networks to combine two images:
- **Content Image** – The subject of the final image.
- **Style Image** – The artistic style to apply to the content.

This implementation is based on the paper [*A Neural Algorithm of Artistic Style* by Gatys et al. (2015)](https://arxiv.org/abs/1508.06576).

## 🚀 Features

- Uses a pre-trained VGG-19 model for feature extraction.
- Separates style and content losses.
- Optimizes input image using L-BFGS algorithm.
- Supports GPU acceleration via CUDA.
- Modular and easy to customize.

## 🧩 Dependencies

- Python 3.x
- PyTorch
- torchvision
- PIL (Pillow)

🛠️ Usage
Clone the repository

bash

git clone https://github.com/your-username/neural-style-transfer-pytorch.git
cd neural-style-transfer-pytorch
Place your images

Replace the default images with your own:

content.jpg – The image whose structure you want to preserve.

style.jpg – The artwork whose style you want to emulate.

Place them in the project directory (or update the paths accordingly in the script).

Run the script

bash

python style_transfer.py
This will:

Load and preprocess the images

Perform the style transfer

Save and display the result as stylized_output.jpg

⚙️ Configuration
You can modify the following parameters in the script:

python

num_steps = 300            # Number of optimization steps
style_weight = 1_000_000   # Importance of style
content_weight = 1         # Importance of content
imsize = (512, 512)        # Resize input images


# Output

![image](https://github.com/user-attachments/assets/42a5e01e-2d42-4cfe-a45c-ce8ccc7fd232)

![image](https://github.com/user-attachments/assets/9c78d351-7dca-4be1-9305-60dd9d742044)

![image](https://github.com/user-attachments/assets/9a62bcc4-6be3-46bf-add5-ccb81426ddcb)







