# SimpleDCGANTest

This repository contains a PyTorch implementation of a Generative Adversarial Network (GAN) for image synthesis. The GAN consists of a generator and a discriminator, and it can be trained on various datasets, such as CIFAR-10, LSUN, MNIST, and ImageNet.

### I use this script to test out new hardware configurations for my AI/ML development rig

## Dependencies

- Python 3
- PyTorch
- torchvision
- tqdm

## Installation:

1. Create a virtual environment (optional, but recommended):

  ```bash
  python -m venv venv
  source venv/bin/activate # On Windows, use: venv\Scripts\activate
  ```
  
2. Install the required packages using `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```

3. After completing the installation, proceed with the [Usage](#usage) section.

## Usage

1. Clone the repository:

```bash
git clone https://github.com/jim-ecker/SimpleDCGANTest.git
cd SimpleDCGANTest
```

2. Train the GAN on a dataset of your choice:

```bash
python main.py --dataset <dataset> --dataroot <path_to_dataset>
```

Replace `<dataset>` with one of the supported datasets (`cifar10`, `lsun`, `mnist`, `imagenet`, `folder`, `lfw`, `fake`) and `<path_to_dataset>` with the path to the dataset on your machine.

3. Check the `./output` folder for generated images and model checkpoints.

## Command-line Arguments

The following command-line arguments can be used to customize the training process:

- `--dataset`: The dataset to train on (default: None, required)
- `--dataroot`: Path to the dataset (default: None, required unless dataset is 'fake')
- `--workers`: Number of data loading workers (default: 2)
- `--batchSize`: Input batch size (default: 128)
- `--imageSize`: Height/width of the input image to the network (default: 64)
- `--nz`: Size of the latent z vector (default: 100)
- `--ngf`: Number of generator filters (default: 64)
- `--ndf`: Number of discriminator filters (default: 64)
- `--niter`: Number of epochs to train for (default: 25)
- `--lr`: Learning rate (default: 0.0002)
- `--beta1`: Beta1 for Adam optimizer (default: 0.5)
- `--cuda`: Enables CUDA (default: False)
- `--dry-run`: Checks if a single training cycle works (default: False)
- `--ngpu`: Number of GPUs to use (default: 1)
- `--netG`: Path to the generator network (to continue training)
- `--netD`: Path to the discriminator network (to continue training)
- `--outf`: Folder to output images and model checkpoints (default: '.')
- `--manualSeed`: Manual seed for random number generation
- `--classes`: Comma-separated list of classes for the LSUN dataset (default: 'bedroom')
- `--mps`: Enables macOS GPU training (default: False)

## Example

To train a model using the CIFAR10 dataset, you can run the following command:

```bash 
python main.py --dataset cifar10 --dataroot <path_to_dataset> --outf <output_folder>
```
Replace `<path_to_dataset>` with the actual path to the CIFAR10 dataset folder and `<output_folder>` with the folder you want the output images and model checkpoints to be saved.

## Outputs

During training, the model will output the following:

1. Real and generated images after every 100 iterations.
2. Model checkpoints (netG and netD) after each epoch.

The generated images and model checkpoints will be saved in the output folder specified by the `--outf` option.

## Acknowledgements

This code is based on the [DCGAN](https://arxiv.org/abs/1511.06434) paper by Radford, Metz, and Chintala. The original PyTorch DCGAN implementation can be found [here](https://github.com/pytorch/examples/tree/master/dcgan).

## License

This project is licensed under the MIT License.
