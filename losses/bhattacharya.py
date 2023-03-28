# Define the loss function for the bhattacharya distance between two normal distributions
########################################################################################################################
# Import dependencies
import torch.nn as nn
import torch
########################################################################################################################
class bhattachayra(nn.Module):
    def __init__(self):
        super(bhattachayra, self).__init__()

    def forward(self, mu1, sigma1, mu2, sigma2):
        '''
        Calculate the bhattachayra distance, only suitable for distributions assumed to be normal
        mu1: torch array of reference source means 
        sigma1: torch array of reference source stds
        mu2: torch array of target means 
        sigma2: torch array of target stds
        return: bhattachayra distance between the two normal distributions
        '''
        assert mu1.size() == sigma1.size() == mu2.size() == sigma2.size()
        # Add small amount to prevent explosion to infinity
        sigma1 = sigma1 + 1e-6
        sigma2 = sigma2 + 1e-6
        squares = torch.pow(sigma1,2) + torch.pow(sigma2,2)
        db = 1/4 * ((mu1 - mu2)**2)/squares + 1/2 * torch.log(squares/(2*sigma1*sigma2))
        return db.mean()
    
class bhattachayra_GMM(nn.Module):
    def __init__(self, n_components, ref_mu, ref_sigma, ref_pi):
        cuda = torch.cuda.is_available()

        super(bhattachayra_GMM, self).__init__() 
        self.n_components = n_components
        if self.n_components not in [1, 2, 3]:
            raise Exception('Only implemented for one, two or three components so far')
        
        self.ref_mu = torch.from_numpy(ref_mu)
        self.ref_sigma = torch.from_numpy(ref_sigma)
        self.ref_pi = torch.from_numpy(ref_pi)
        self.loss = bhattachayra()
        
        if cuda:
            self.ref_mu = self.ref_mu.cuda()
            self.ref_sigma = self.ref_sigma.cuda()
            self.ref_pi = self.ref_pi.cuda()

        assert(ref_mu.shape[1] == self.n_components)

    def forward(self, mus, sigmas, pis):
        '''
        Calculate the bhattachayra distance summed up, only suitable for distributions assumed to be normal
        mus: torch vector of reference target means
        sigmas: torch vector of reference target stds
        pis: torch vector of reference target pis
        return: bhattachayra distance between the two GMMs
        '''
        mus = mus.squeeze()
        sigmas = sigmas.squeeze()
        pis = pis.squeeze()
        
        if self.n_components == 1:
            mus = mus.view(-1, 1)
            sigmas = sigmas.view(-1, 1)
            pis = pis.view(-1, 1)
            
        assert mus.shape == self.ref_mu.shape
        assert sigmas.shape == self.ref_sigma.shape        
        assert pis.shape == self.ref_pi.shape
        
        if self.n_components == 1:
            dist00 = self.ref_pi[:]* pis[:]*self.loss(self.ref_mu[:], self.ref_sigma[:], mus[:], sigmas[:])
            total = dist00
        if self.n_components == 2:
            dist00 = self.ref_pi[:,0]* pis[:,0]*self.loss(self.ref_mu[:,0], self.ref_sigma[:,0], mus[:,0], sigmas[:,0])
            dist11 = self.ref_pi[:,1]* pis[:,1]*self.loss(self.ref_mu[:,1], self.ref_sigma[:,1], mus[:,1], sigmas[:,1])
            total = dist00  + dist11
        if self.n_components == 3: 
            dist00 = self.ref_pi[:,0]* pis[:,0]*self.loss(self.ref_mu[:,0], self.ref_sigma[:,0], mus[:,0], sigmas[:,0])
            dist11 = self.ref_pi[:,1]* pis[:,1]*self.loss(self.ref_mu[:,1], self.ref_sigma[:,1], mus[:,1], sigmas[:,1])
            dist22 = self.ref_pi[:,2]* pis[:,2]*self.loss(self.ref_mu[:,2], self.ref_sigma[:,2], mus[:,2], sigmas[:,2])
            total = dist00 + dist11 + dist22
        return total.mean()
    
    
