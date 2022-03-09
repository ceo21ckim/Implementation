import torch 
import torch.nn as nn 
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import os


# !pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

torch.__version__


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# hyper-parameters
latent_size = 64
hidden_size = 256
image_size = 784
num_epochs = 200
batch_size = 100
sample_dir = 'samples'

# Create a directory if not exists
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
    
# Image processing
# transform = transforms.Compose([
#     transforms.Totensor(),
#     transforms.Normalize(mean = (0.5, 0.5, 0.5),
#                         std = (0.5, 0.5, 0.5))
# ])3 for RGB channels

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = 0.5, std = 0.5)
])


# MNIST dataset 
mnist = torchvision.datasets.MNIST(root = '../../data/',
                                  train= True, 
                                  transform = transform,
                                  download = True)

# Data loader

data_loader = DataLoader(dataset = mnist, 
                        batch_size = batch_size, 
                        shuffle = True)

# Discriminator
D = nn.Sequential(
nn.Linear(image_size, hidden_size), 
nn.LeakyReLU(0.2),
nn.Linear(hidden_size, hidden_size),
nn.LeakyReLU(0.2),
nn.Linear(hidden_size, 1),
nn.Sigmoid()
)

# Generator
G = nn.Sequential(
nn.Linear(latent_size, hidden_size),
nn.ReLU(),
nn.Linear(hidden_size, hidden_size),
nn.ReLU(),
nn.Linear(hidden_size, image_size),
nn.Tanh()
)

# Device setting
D = D.to(device)
G = G.to(device)

# Binary cross entropy loss and optimizer 
criterion = nn.BCELoss()
d_optimizer = optim.Adam(D.parameters(), lr = 0.0002)
g_optimizer = optim.Adam(G.parameters(), lr = 0.0002)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1) # 0보다 작으면 0,  1보다 큰 값은 1로 변환해준다. 


def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()


# Start training
total_step = len(data_loader)

for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        images = images.reshape(batch_size, -1).to(device)
        
        # Create the labels which are later used as input for the BCE loss
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # ============================================================= #
        #                  Train the discriminator                      #
        # ============================================================= #
        
        # Compute BCE_Loss using real images where BCE_Loss(x, y) : - y * log(D(x)) - (1 - y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        
        real_score = outputs 
        
        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0 
        
        z = torch.randn(batch_size, latent_size).to(device)
        fake_image = G(z)
        outputs = D(fake_image)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs
        
        # Backprop and optimize
        d_loss = d_loss_real + d_loss_fake
        reset_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # ============================================================= #
        #                  Train the Generator                          #
        # ============================================================= #
        
        # Compute loss with fake images
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        
        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z))))
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
        
        g_loss = criterion(outputs, real_labels)
        
        # Backprop and optimize
        reset_grad()
        g_loss.backward()
        g_optimizer.step()
        
        # Save real images
        if (i+1) % 200 == 1:
            images = images.reshape(images.size(0), 1, 28, 28)
            save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'))
            
        # Save sampled images 
        fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
        save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))
        

# Save the model checkpoints
torch.save(G.state_dict(), 'G.chpt')
torch.save(D.state_dict(), 'D.chpt')