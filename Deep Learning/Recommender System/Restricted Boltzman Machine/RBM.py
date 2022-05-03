# 원 출처 : https://github.com/odie2630463/Restricted-Boltzmann-Machines-in-pytorch

import numpy as np 
import torch, os
import torch.utils.data 
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 

from torch.autograd import Variable 
from torchvision import datasets, transforms 
from torchvision.utils import make_grid 
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
import matplotlib.pyplot as plt 

torch.cuda.set_device('cuda:0')

torch.cuda.current_device()
torch.cuda.get_device_name('cuda')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
def show_adn_save(file_name, img):
    # 이미지를 출력해주는 함수 
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    f = './%s.png' % file_name
    # plt.imshow(npimg)
    plt.imsave(f, npimg)


class RBM(nn.Module):
    def __init__(self, n_visible=784, n_hidden=500, k=5): # n_visible : visible unit / n_hidden : hidden unit 
        super(RBM, self).__init__()

        # 만약 nn.Parameter 혹은 Variable로 변수를 생성할 시 device를 지정해주어야 함...
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible, device=device)*1e-2) # 500 by 784
        self.v_bias = nn.Parameter(torch.zeros(n_visible, device=device)) # 초기 bias는 0으로 설정함.
        self.h_bias = nn.Parameter(torch.zeros(n_hidden, device=device)) # 초기 bias는 0으로 설정함.

        self.k = k # Contrastive Divergence 의 횟수
        """
        Contrastive Divergence 는 Gibbs sampling의 step을 converge할 때까지 돌리는 것이 아니라, 
        한 번만 돌려 p(v,h)를 approximate하고, 해당 값을 이용해 sum(p(v,b)*v*h)를 계산하여 grdient의 approximation 값을 구하는 방식
        """

    def sample_from_p(self, p):
        '''
        Gibbs Sampling 과정 
        한 번의 p(v, h) 확률 값을 가지고 uniform distribution으로부터 얻은 값보다 크면 1, 아니면 -1로 출력함.
        그 후 ReLU를 통해 -1로 출력된 값은 0을 출력하도록 설정함.
        '''
        _p = p - Variable(torch.rand(p.size(), device=device))
        p_sign = torch.sign(_p)
        return F.relu(p_sign)

    def v_to_h(self, v):
        # 주어진 visible units로 부터 hidden units을 sampling하는 과정
        p_h = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        sample_h = self.sample_from_p(p_h)
        return p_h, sample_h

    def h_to_v(self, h):
        # hidden unit의 샘플로부터 visible unit을 다시 복원하는 과정
        p_v = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        sample_v = self.sample_from_p(p_v)
        return p_v, sample_v

    def forward(self, v):
        # visible -> hidden -> visible
        pre_h1, h1 = self.v_to_h(v)

        h_ = h1

        for _ in range(self.k): # Contrastive Divergence 를 k번 수행함.
            pre_v_, v_ = self.h_to_v(h_)
            pre_h_, h_ = self.v_to_h(v_)


            # v는 입력으로 받은 원래 image, v_는 sampling으로 얻은 h로 부터 다시 획득한 sample
            return v, v_ 

    def free_energy(self,v):
        # free energe 계산
        vbias_term = v.mv(self.v_bias) # mv : matrix - vector product
        wx_b = F.linear(v, self.W, self.h_bias)

        temp = torch.log(torch.exp(wx_b) + 1)
        hidden_term = torch.sum(temp, dim = 1)

        return (-hidden_term - vbias_term).mean()


if __name__ == '__main__':

# configuration arguments
    batch_size = 256
    epochs = 10
    transforms_tensor = transforms.Compose([transforms.ToTensor()])

    # load dataset : MNIST
    train_loader = DataLoader( datasets.MNIST('./MNIST_data', train=True, download=True, transform=transforms_tensor), batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader( datasets.MNIST('./MNIST_data', train=False, transform=transforms_tensor), batch_size=batch_size)


    rbm = RBM(k=5).to(device)

    optimization = optim.Adam(rbm.parameters(), lr=0.005)

    rbm.train()
    for epoch in range(epochs):
        loss_ = []
        for (data, target) in tqdm(train_loader) :
            data = Variable(data.view(-1, 784)) # 28 by 28 이미지를 784로 변환함.
            sample_data = data.bernoulli() # RBM의 입력은 0또는 1의 값만 가짐.
            sample_data = sample_data.to(device)
            v, v1 = rbm(sample_data) # v :원본 이미지, v1 : 카피한 이미지
            v = v.to(device)
            v1 = v1.to(device)

            loss = rbm.free_energy(v) - rbm.free_energy(v1)
            loss_.append(loss.data.item())
            optimization.zero_grad()
            loss.backward()

            optimization.step()

        print(np.mean(loss_))

    testset = datasets.MNIST('./MNIST_data', train=False, transform=transforms_tensor)

    sample_data = testset.data[:32, :].view(-1,784)
    sample_data = sample_data.type(torch.FloatTensor)/255
    sample_data = sample_data.to(device)
    v, v1 = rbm(sample_data)
    show_adn_save('output/real_testdata', make_grid(v.view(32, 1, 28, 28).data).cpu())
    show_adn_save('output/generated_testdata', make_grid(v1.view(32, 1, 28, 28).data).cpu())


