# DCGAN Implementation

This repository contains an implementation of Deep Convolutional Generative Adversarial Networks (DCGAN) for generating synthetic images. The model is trained on the CelebA dataset, producing human-like face images.

## Dataset Preprocessing

1. **Dataset Used**: CelebA dataset
2. **Steps to Prepare the Dataset**:
   - Download the **img_align_celeba** dataset from the official CelebA dataset website.
   - Place all images in `data/celeba/img_align_celeba`.
   - Ensure there are no subdirectories.
   - The dataset is loaded using a custom `Dataset` class.
   - Images are resized to **64x64** and normalized to **[-1, 1]** using:
     ```python
     transform = transforms.Compose([
         transforms.Resize(64),
         transforms.CenterCrop(64),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])
     ```
   - Use PyTorch's `DataLoader` for efficient loading:
     ```python
     dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
     ```

## Training the Model

1. Run the training script:
   ```bash
   python train.py
   ```
2. **Training Parameters**:
   - Number of epochs: **10** (modifiable)
   - Batch size: **64**
   - Latent vector (noise) dimension: **100**
   - Adam optimizer with **lr=0.0002**, **beta1=0.5**
   - The generator is trained **3 times** for every discriminator update to balance learning.
3. **Training Outputs**:
   - The console prints training progress:
     ```
     Epoch [1/10]:  [200/3166] Loss_D: 0.75, Loss_G: 2.56
     ```
   - The model checkpoints are saved after every epoch in `checkpoints/`:
     ```
     checkpoints/
       ├── netG_epoch_1.pth
       ├── netD_epoch_1.pth
       ├── ...
     ```

## Testing the Model

1. To generate images from a trained model:
   ```python
   python generate.py --model checkpoints/netG_epoch_10.pth
   ```
2. The model takes random noise as input and outputs generated images.
3. Images are saved in the `generated_samples/` directory.

## Comparing Real vs Generated Images

- The script plots a side-by-side comparison of **real vs. generated images** after training:
  ```python
  def compare_images(real_batch, fake_batch):
      fig, axes = plt.subplots(2, 8, figsize=(12, 4))
      for i in range(8):
          axes[0, i].imshow(real_batch[i].permute(1, 2, 0).cpu().numpy())
          axes[0, i].axis('off')
          axes[1, i].imshow(fake_batch[i].permute(1, 2, 0).cpu().numpy())
          axes[1, i].axis('off')
      plt.show()
  ```

## Results

- The generator improves over epochs, producing realistic images.
- The discriminator loss should ideally stabilize while the generator keeps improving.
- The final output includes high-quality synthetic faces similar to the CelebA dataset.

## Dependencies
Ensure the following Python packages are installed:
```bash
pip install torch torchvision matplotlib tqdm
```

## Notes
- If the dataset is not found, verify that images are directly in `img_align_celeba/`.
- For training stability, ensure batch normalization is applied in both networks.

---

This completes the README documentation. You can modify it as per your needs.

