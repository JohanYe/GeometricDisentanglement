import torch
import torch.nn as nn
from geoml import *
import geoml.nnj as nnj
from torch.nn.functional import softplus
import torch.distributions as td
from torch.distributions.kl import kl_divergence as KL
import matplotlib

class BasicVAE(nn.Module):
    def __init__(self, hidden_layer=[512, 256], latent_space=2):
        super(BasicVAE, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_layer = hidden_layer
        self.latent_space = latent_space
        self.prior_loc = torch.zeros(latent_space, device=self.device)
        self.prior_scale = torch.ones(latent_space, device=self.device)
        self.prior = td.Independent(td.Normal(loc=self.prior_loc, scale=self.prior_scale), 1)

        enc = [nnj.ResidualBlock(nnj.Linear(784, hidden_layer[0]), nnj.Softplus())]
        for i in range(len(hidden_layer) - 1):
            enc.append(nnj.ResidualBlock(nnj.Linear(hidden_layer[i], hidden_layer[i + 1]), nnj.Softplus()))
        enc.append(nnj.Linear(hidden_layer[-1], int(latent_space * 2)))
        self.encoder = nnj.Sequential(*enc)

        dec = [nnj.Linear(latent_space, hidden_layer[-1]), nnj.Softplus()]
        for i in reversed(range(1, len(hidden_layer))):
            dec.append(nnj.ResidualBlock(nnj.Linear(hidden_layer[i], hidden_layer[i - 1]), nnj.Softplus()))
        dec.extend([nnj.ResidualBlock(nnj.Linear(hidden_layer[0], 784), nnj.Sigmoid())])
        self.decoder_loc = nnj.Sequential(*dec)
        #         self.decoder_loc = nnj.Sequential(nnj.ResidualBlock(nnj.Linear(hidden_layer[0], 784), nnj.Sigmoid()))

        self.init_decoder_scale = 0.01 * torch.ones(784, device=self.device)
        self.decoder_std = None

    def decoder_scale(self, z):
        if self.decoder_std is None:
            return self.init_decoder_scale
        else:
            return self.decoder_std(z)

    def encode(self, x):
        z_mu, z_lv = torch.chunk(self.encoder(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=z_mu, scale=softplus(z_lv) + 1e-10), 1)

    def decode(self, z):
        #         out = self.decoder(z)
        x_mu = self.decoder_loc(z)
        x_std = self.decoder_scale(z)
        return td.Independent(td.Normal(loc=x_mu, scale=x_std + 1e-10), 1)

    def elbo(self, x, kl_weight=1):
        q_zGx = self.encode(x)
        z = q_zGx.rsample()
        p_xGz = self.decode(z)
        ELBO = torch.mean(p_xGz.log_prob(x) - kl_weight * KL(q_zGx, self.prior), dim=0)
        return ELBO

    def fit_mean(self, data_loader, num_epochs=150, num_cycles=30):
        switch_epoch = num_epochs // 3
        if self.decoder_std is not None:
            for layer in self.parameters():
                layer.requires_grad = True
            for layer in self.decoder_std.parameters():
                layer.requires_grad = False

        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        train_loss = []

        for cycle in range(num_cycles):
            for epoch in range(num_epochs):
                sum_loss = 0
                # implements cyclic KL scaling
                lam = min(1.0, max(0.0, 2.0 * (epoch - switch_epoch) / (num_epochs - switch_epoch)))
                for batch_idx, (data,) in enumerate(data_loader):
                    data = data.to(self.device)
                    optimizer.zero_grad()
                    loss = -self.elbo(data, lam)
                    loss.backward()
                    optimizer.step()
                    sum_loss += loss.item() * len(data)
                avg_loss = sum_loss / len(data_loader.dataset)
                train_loss.append(avg_loss)
                print('(MEAN; lambda={:.4f}) ====> Epoch: {} Average loss: {:.4f}'.format(lam, epoch, avg_loss))

        return train_loss

    def init_std_naive(self):
        #         self.decoder_std = nnj.Sequential(*[nnj.ResidualBlock(nnj.Linear(self.hidden_layer[0], 784)),
        #                                            nnj.Sigmoid()]).to(self.device)
        dec = [nnj.Linear(self.latent_space, self.hidden_layer[-1]), nnj.Softplus()]
        for i in reversed(range(1, len(self.hidden_layer))):
            dec.append(nnj.ResidualBlock(nnj.Linear(self.hidden_layer[i], self.hidden_layer[i - 1]), nnj.Softplus()))
        dec.extend([nnj.Linear(self.hidden_layer[0], 784), nnj.Softplus()])
        self.decoder_std = nnj.Sequential(*dec).to(self.device)

    def fit_std(self, data_loader, num_epochs=150):
        for i in self.parameters():
            i.requires_grad = False
        for i in self.decoder_std.parameters():
            i.requires_grad = True
        optimizer = torch.optim.Adam(self.decoder_std.parameters(), weight_decay=2e-4)  # Only backprop over std
        std_train = []
        for epoch in range(num_epochs):
            sum_loss = 0
            for batch_idx, (data,) in enumerate(data_loader):
                data = data.to(self.device)
                optimizer.zero_grad()
                loss = -self.elbo(data, 1)
                loss.backward()
                sum_loss += loss.item() * len(data)
                optimizer.step()
            avg_loss = sum_loss / len(data_loader.dataset)
            std_train.append(avg_loss)
            print('(STD)  ====> Epoch: {} Average loss: {:.4f}'.format(epoch, avg_loss))

        return std_train


class VAE_dsprites(nn.Module, EmbeddedManifold):
    def __init__(self, x, layers, latents_bases=None, latents_sizes=None, num_components=100, device=None):
        super(VAE_dsprites, self).__init__()

        self.device = device
        self.latents_bases = latents_bases
        self.latents_sizes = latents_sizes

        self.p = int(layers[0])  # Dimension of x
        self.d = int(layers[-1])  # Dimension of z
        self.h = layers  # [1:-1] # Dimension of hidden layers
        self.num_components = num_components

        enc = []
        for k in range(len(layers) - 1):
            in_features = int(layers[k])
            out_features = int(layers[k + 1])
            enc.append(nnj.ResidualBlock(nnj.Linear(in_features, out_features),
                                         nnj.Softplus()))
        enc.append(nnj.Linear(out_features, int(self.d * 2)))

        dec = []
        for k in reversed(range(len(layers) - 1)):
            in_features = int(layers[k + 1])
            out_features = int(layers[k])
            dec.append(nnj.ResidualBlock(nnj.Linear(in_features, out_features),
                                         nnj.Softplus()))
        dec.append(nnj.Sigmoid())

        # Note how we use 'nnj' instead of 'nn' -- this gives automatic
        # computation of Jacobians of the implemented neural network.
        # The embed function is required to also return Jacobians if
        # requested; by using 'nnj' this becomes a trivial constraint.
        self.encoder = nnj.Sequential(*enc)

        self.decoder_loc = nnj.Sequential(*dec)
        self.init_decoder_scale = 0.01 * torch.ones(self.p, device=self.device)

        self.prior_loc = torch.zeros(self.d, device=self.device)
        self.prior_scale = torch.ones(self.d, device=self.device)
        self.prior = td.Independent(td.Normal(loc=self.prior_loc, scale=self.prior_scale), 1)

        # Create a blank std-network.
        # It is important to call init_std after training the mean, but before training the std
        self.dec_std = None

        self.to(self.device)

    def encode(self, x):
        z_loc, z_scale = torch.chunk(self.encoder(x), chunks=2, dim=-1)
        return td.Independent(td.Normal(loc=z_loc, scale=z_scale.mul(0.5).exp() + 1e-10), 1)

    def decode(self, z):
        x_loc = self.decoder_loc(z)
        # return td.Independent(td.Normal(loc=x_loc, scale=self.decoder_scale(z) + 1e-10), 1)
        return td.Normal(loc=x_loc, scale=self.decoder_scale(z) + 1e-10)

    def decoder_scale(self, z, jacobian=False):
        if self.dec_std is None:
            return self.init_decoder_scale
        else:
            return self.dec_std(z, jacobian=jacobian)

    def elbo(self, x, kl_weight=1.0):
        q = self.encode(x)
        z = q.rsample()  # (batch size)x(latent dim)
        px_z = self.decode(z)  # p(x|z)
        ELBO = px_z.log_prob(x).mean() - kl_weight * KL(q, self.prior).mean()
        return ELBO

    def latent_to_index(self, latents):
        return np.dot(latents, self.latents_bases).astype(int)

    def sample_latent(self, size=1):
        samples = np.zeros((size, self.latents_sizes.size))
        for lat_i, lat_size in enumerate(self.latents_sizes):
            if lat_i in [0, 1, 3]:  # if color or shape
                samples[:, lat_i] = 0
            else:
                samples[:, lat_i] = np.random.randint(lat_size, size=size)
        return samples

    def fit_mean(self, images, num_epochs=150, num_cycles=30, max_kl=2, batch_size=256, sampling_per_epoch=24):
        switch_epoch = num_epochs // 5
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        for cycle in range(1, num_cycles + 1):
            print('Cycle:', cycle)
            for epoch in range(num_epochs):
                sum_loss = 0
                # implements cyclic KL scaling
                if cycle < num_cycles - num_cycles // 5:
                    lam = min(max_kl, max(0.0, 2 * max_kl * (epoch - switch_epoch) / (num_epochs - switch_epoch)))
                else:
                    lam = 1
                for i in range(sampling_per_epoch):
                    data_idx = self.latent_to_index(self.sample_latent(size=batch_size))  #
                    data = torch.Tensor(images[data_idx]).to(self.device)
                    data = data.reshape(data.shape[0], int(data.shape[1] * data.shape[2]))
                    optimizer.zero_grad()
                    loss = -self.elbo(data, lam)
                    loss.backward()
                    optimizer.step()
                    sum_loss += loss.item() * len(data)
                avg_loss = sum_loss / (sampling_per_epoch * batch_size)
                print('(MEAN; lambda={:.4f}) ====> Epoch: {} Average loss: {:.4f}'.format(lam, epoch, avg_loss))

    # This function sets up the data structures for the RBF network for modeling variance.
    # XXX: We should do this more elagantly directly in __init__
    def init_std(self, x, gmm_mu=None, gmm_cv=None, weights=None):
        with torch.no_grad():
            z = torch.chunk(self.encoder(x.to(self.device)), chunks=2, dim=-1)[0]
        N, D = x.shape
        d = z.shape[1]
        inv_maxstd = 1e-1  # 1.0 / x.std(dim=0).mean() # x.std(dim=0).mean() #D*x.var(dim=0).mean()

        if gmm_mu is None and gmm_cv is None and weights is None:
            from sklearn import mixture
            clf = mixture.GaussianMixture(n_components=self.num_components, covariance_type='spherical')
            clf.fit(z.cpu().numpy())
            self.gmm_means = clf.means_
            self.gmm_covariances = clf.covariances_
            self.clf_weights = clf.weights_
        else:
            print('loading weights...')
            self.gmm_means = gmm_mu
            self.gmm_covariances = gmm_cv
            self.clf_weights = weights

        self.dec_std = nnj.Sequential(nnj.RBF(d, self.num_components,
                                              points=torch.tensor(self.gmm_means, dtype=torch.float,
                                                                  requires_grad=False),
                                              beta=0.5 / torch.tensor(self.gmm_covariances, dtype=torch.float,
                                                                      requires_grad=False)),  # d --> num_components
                                      nnj.PosLinear(self.num_components, 1, bias=False),  # num_components --> 1
                                      nnj.Reciprocal(inv_maxstd),  # 1 --> 1
                                      nnj.PosLinear(1, D)).to(self.device)  # 1 --> D
        with torch.no_grad():
            self.dec_std[1].weight[:] = ((torch.tensor(self.clf_weights, dtype=torch.float).exp() - 1.0).log()).to(
                self.device)
        self.dec_std

    def fit_std(self, images, num_epochs, sampling_per_epoch=24, batch_size=256):
        optimizer = torch.optim.Adam(self.dec_std.parameters(), weight_decay=1e-4)
        for epoch in range(num_epochs):
            sum_loss = 0
            for i in range(sampling_per_epoch):
                data_idx = self.latent_to_index(self.sample_latent(size=batch_size))  #
                data = torch.Tensor(images[data_idx]).to(self.device)
                data = data.reshape(data.shape[0], int(data.shape[1] * data.shape[2]))
                optimizer.zero_grad()
                # XXX: we should sample here
                loss = -self.elbo(data, 1)
                loss.backward()
                sum_loss += loss.item() * len(data)
                optimizer.step()
            avg_loss = sum_loss / sampling_per_epoch
            print('(STD)  ====> Epoch: {} Average loss: {:.4f}'.format(epoch, avg_loss))

    def plot(self, z, labels, meshsize=150):
        import matplotlib.pyplot as plt
        if z.shape[1] == 2:
            minz, _ = z.min(dim=0)  # d
            maxz, _ = z.max(dim=0)  # d
            alpha = 0.1 * (maxz - minz)  # d
            minz -= alpha  # d
            maxz += alpha  # d

            ran0 = torch.linspace(minz[0].item(), maxz[0].item(), meshsize, device=self.device)
            ran1 = torch.linspace(minz[1].item(), maxz[1].item(), meshsize, device=self.device)
            Mx, My = torch.meshgrid(ran0, ran1)
            Mxy = torch.cat((Mx.t().reshape(-1, 1), My.t().reshape(-1, 1)), dim=1)  # (meshsize^2)x2
            with torch.no_grad():
                varim = self.decoder_scale(Mxy).pow(2).mean(dim=-1).reshape(meshsize, meshsize)
                varim = varim.cpu().detach().numpy()
            plt.imshow(varim, extent=(ran0[0].item(), ran0[-1].item(), ran1[0].item(), ran1[-1].item()), origin='lower')
            plt.colorbar()

            for label in labels.unique():
                idx = labels == label
                zi = z[idx].cpu().detach().numpy()
                plt.plot(zi[:, 0], zi[:, 1], '.')
        else:
            raise Exception('Latent dimension not suitable for plotting')

    def embed(self, points, jacobian=False):
        """
        Embed the manifold into (mu, std) space.

        Input:
            points:     a Nx(d) or BxNx(d) torch Tensor representing a (batch of a)
                        set of N points in latent space that will be embedded
                        in R^2D.

        Optional input:
            jacobian:   a boolean indicating if the Jacobian matrix of the function
                        should also be returned. Default is False.

        Output:
            embedded:   a Nx(2D) of BxNx(2D) torch tensor containing the N embedded points.
                        The first Nx(d) part contain the mean part of the embedding,
                        whlie the last Nx(d) part contain the standard deviation
                        embedding.

        Optional output:
            J:          If jacobian=True then a second Nx(2D)x(d) or BxNx(2D)x(d)
                        torch tensor is returned that contain the Jacobian matrix
                        of the embedding function.
        """
        std_scale = 1.0
        is_batched = points.dim() > 2
        if not is_batched:
            points = points.unsqueeze(0)  # BxNxD

        if jacobian:
            mu, Jmu = self.decoder_loc(points, jacobian=True)  # BxNxD, BxNxDx(d)
            std, Jstd = self.decoder_scale(points, jacobian=True)  # BxNxD, BxNxDx(d)
            embedded = torch.cat((mu, std_scale * std), dim=2)  # BxNx(2D)
            J = torch.cat((Jmu, std_scale * Jstd), dim=2)  # BxNx(2D)x(d)
        else:
            mu = self.decoder_loc(points)  # BxNxD
            std = self.decoder_scale(points)  # BxNxD
            embedded = torch.cat((mu, std_scale * std), dim=2)  # BxNx(2D)

        if not is_batched:
            embedded = embedded.squeeze(0)
            if jacobian:
                J = J.squeeze(0)

        if jacobian:
            return embedded, J
        else:
            return embedded

class VAE_3dshapes(nn.Module, EmbeddedManifold):
    def __init__(self, layers, latents_bases=None, latents_sizes=None, num_components=100, device=None):
        super(VAE_3dshapes, self).__init__()

        self.device = device
        self.latents_bases = latents_bases
        self.latents_sizes = latents_sizes

        self.p = int(layers[0])  # Dimension of x
        self.d = int(layers[-1])  # Dimension of z
        self.h = layers  # [1:-1] # Dimension of hidden layers
        self.num_components = num_components

        enc = []
        for k in range(len(layers) - 1):
            in_features = int(layers[k])
            out_features = int(layers[k + 1])
            enc.append(nnj.ResidualBlock(nnj.Linear(in_features, out_features),
                                         nnj.Softplus()))
        enc.append(nnj.Linear(out_features, int(self.d * 2)))

        dec = []
        for k in reversed(range(len(layers) - 1)):
            in_features = int(layers[k + 1])
            out_features = int(layers[k])
            dec.append(nnj.ResidualBlock(nnj.Linear(in_features, out_features),
                                         nnj.Softplus()))
        dec.append(nnj.Sigmoid())

        # Note how we use 'nnj' instead of 'nn' -- this gives automatic
        # computation of Jacobians of the implemented neural network.
        # The embed function is required to also return Jacobians if
        # requested; by using 'nnj' this becomes a trivial constraint.
        self.encoder = nnj.Sequential(*enc)

        self.decoder_loc = nnj.Sequential(*dec)
        self.init_decoder_scale = 0.01 * torch.ones(self.p, device=self.device)

        self.prior_loc = torch.zeros(self.d, device=self.device)
        self.prior_scale = torch.ones(self.d, device=self.device)
        self.prior = td.Independent(td.Normal(loc=self.prior_loc, scale=self.prior_scale), 1)

        # Create a blank std-network.
        # It is important to call init_std after training the mean, but before training the std
        self.dec_std = None

        self.to(self.device)

    def encode(self, x):
        z_loc, z_scale = torch.chunk(self.encoder(x), chunks=2, dim=-1)
        return td.Independent(td.Normal(loc=z_loc, scale=z_scale.mul(0.5).exp() + 1e-10), 1)

    def decode(self, z):
        x_loc = self.decoder_loc(z)
        # return td.Independent(td.Normal(loc=x_loc, scale=self.decoder_scale(z) + 1e-10), 1)
        return td.Normal(loc=x_loc, scale=self.decoder_scale(z) + 1e-10)

    def decoder_scale(self, z, jacobian=False):
        if self.dec_std is None:
            return self.init_decoder_scale
        else:
            return self.dec_std(z, jacobian=jacobian)

    def elbo(self, x, kl_weight=1.0):
        q = self.encode(x)
        z = q.rsample()  # (batch size)x(latent dim)
        px_z = self.decode(z)  # p(x|z)
        ELBO = px_z.log_prob(x).mean() - kl_weight * KL(q, self.prior).mean()
        return ELBO

    def fit_mean(self, data_loader, num_epochs=150, num_cycles=30, max_kl=5):
        switch_epoch = num_epochs // 5
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        for cycle in range(1, num_cycles + 1):
            print('Cycle:', cycle)
            for epoch in range(num_epochs):
                sum_loss = 0
                # implements cyclic KL scaling
                if cycle < num_cycles - num_cycles // 5:
                    lam = min(max_kl, max(0.0, 2 * max_kl * (epoch - switch_epoch) / (num_epochs - switch_epoch)))
                else:
                    lam = 1
                for batch_idx, data in enumerate(data_loader):
                    data = data[0][:,0,:,:]
                    data = data.squeeze(1).to(self.device)
                    data = data.reshape(data.shape[0],int(data.shape[1]*data.shape[2]))
                    optimizer.zero_grad()
                    loss = -self.elbo(data, lam)
                    loss.backward()
                    optimizer.step()
                    sum_loss += loss.item() * len(data)
                avg_loss = sum_loss / len(data_loader.dataset)
                print('(MEAN; lambda={:.4f}) ====> Epoch: {} Average loss: {:.4f}'.format(lam, epoch, avg_loss))

    # This function sets up the data structures for the RBF network for modeling variance.
    # XXX: We should do this more elagantly directly in __init__
    def init_std(self, x, gmm_mu=None, gmm_cv=None, weights=None):
        with torch.no_grad():
            z = torch.chunk(self.encoder(x.to(self.device)), chunks=2, dim=-1)[0]
        N, D = x.shape
        d = z.shape[1]
        inv_maxstd = 1e-1  # 1.0 / x.std(dim=0).mean() # x.std(dim=0).mean() #D*x.var(dim=0).mean()

        if gmm_mu is None and gmm_cv is None and weights is None:
            from sklearn import mixture
            clf = mixture.GaussianMixture(n_components=self.num_components, covariance_type='spherical')
            clf.fit(z.cpu().numpy())
            self.gmm_means = clf.means_
            self.gmm_covariances = clf.covariances_
            self.clf_weights = clf.weights_
        else:
            print('loading weights...')
            self.gmm_means = gmm_mu
            self.gmm_covariances = gmm_cv
            self.clf_weights = weights

        self.dec_std = nnj.Sequential(nnj.RBF(d, self.num_components,
                                              points=torch.tensor(self.gmm_means, dtype=torch.float,
                                                                  requires_grad=False),
                                              beta=0.5 / torch.tensor(self.gmm_covariances, dtype=torch.float,
                                                                      requires_grad=False)),  # d --> num_components
                                      nnj.PosLinear(self.num_components, 1, bias=False),  # num_components --> 1
                                      nnj.Reciprocal(inv_maxstd),  # 1 --> 1
                                      nnj.PosLinear(1, D)).to(self.device)  # 1 --> D
        with torch.no_grad():
            self.dec_std[1].weight[:] = ((torch.tensor(self.clf_weights, dtype=torch.float).exp() - 1.0).log()).to(
                self.device)
        self.dec_std

    def fit_std(self, data_loader, num_epochs, sampling_per_epoch=24, batch_size=256):
        optimizer = torch.optim.Adam(self.dec_std.parameters(), weight_decay=1e-4)
        for epoch in range(num_epochs):
            sum_loss = 0
            for batch_idx, data in enumerate(data_loader):
                data = data[0][:, 0, :, :]
                data = data.squeeze(1).to(self.device)
                data = data.reshape(data.shape[0], int(data.shape[1] * data.shape[2]))
                optimizer.zero_grad()
                # XXX: we should sample here
                loss = -self.elbo(data, 1)
                loss.backward()
                sum_loss += loss.item() * len(data)
                optimizer.step()
            avg_loss = sum_loss / sampling_per_epoch
            print('(STD)  ====> Epoch: {} Average loss: {:.4f}'.format(epoch, avg_loss))

    def plot(self, z, labels, meshsize=150):
        import matplotlib.pyplot as plt
        if z.shape[1] == 2:
            minz, _ = z.min(dim=0)  # d
            maxz, _ = z.max(dim=0)  # d
            alpha = 0.1 * (maxz - minz)  # d
            minz -= alpha  # d
            maxz += alpha  # d

            ran0 = torch.linspace(minz[0].item(), maxz[0].item(), meshsize, device=self.device)
            ran1 = torch.linspace(minz[1].item(), maxz[1].item(), meshsize, device=self.device)
            Mx, My = torch.meshgrid(ran0, ran1)
            Mxy = torch.cat((Mx.t().reshape(-1, 1), My.t().reshape(-1, 1)), dim=1)  # (meshsize^2)x2
            with torch.no_grad():
                varim = self.decoder_scale(Mxy).pow(2).mean(dim=-1).reshape(meshsize, meshsize)
                varim = varim.cpu().detach().numpy()
            plt.imshow(varim, extent=(ran0[0].item(), ran0[-1].item(), ran1[0].item(), ran1[-1].item()), origin='lower')
            plt.colorbar()

            for label in labels.unique():
                idx = labels == label
                zi = z[idx].cpu().detach().numpy()
                plt.plot(zi[:, 0], zi[:, 1], '.')
        else:
            raise Exception('Latent dimension not suitable for plotting')

    def embed(self, points, jacobian=False):
        """
        Embed the manifold into (mu, std) space.

        Input:
            points:     a Nx(d) or BxNx(d) torch Tensor representing a (batch of a)
                        set of N points in latent space that will be embedded
                        in R^2D.

        Optional input:
            jacobian:   a boolean indicating if the Jacobian matrix of the function
                        should also be returned. Default is False.

        Output:
            embedded:   a Nx(2D) of BxNx(2D) torch tensor containing the N embedded points.
                        The first Nx(d) part contain the mean part of the embedding,
                        whlie the last Nx(d) part contain the standard deviation
                        embedding.

        Optional output:
            J:          If jacobian=True then a second Nx(2D)x(d) or BxNx(2D)x(d)
                        torch tensor is returned that contain the Jacobian matrix
                        of the embedding function.
        """
        std_scale = 1.0
        is_batched = points.dim() > 2
        if not is_batched:
            points = points.unsqueeze(0)  # BxNxD

        if jacobian:
            mu, Jmu = self.decoder_loc(points, jacobian=True)  # BxNxD, BxNxDx(d)
            std, Jstd = self.decoder_scale(points, jacobian=True)  # BxNxD, BxNxDx(d)
            embedded = torch.cat((mu, std_scale * std), dim=2)  # BxNx(2D)
            J = torch.cat((Jmu, std_scale * Jstd), dim=2)  # BxNx(2D)x(d)
        else:
            mu = self.decoder_loc(points)  # BxNxD
            std = self.decoder_scale(points)  # BxNxD
            embedded = torch.cat((mu, std_scale * std), dim=2)  # BxNx(2D)

        if not is_batched:
            embedded = embedded.squeeze(0)
            if jacobian:
                J = J.squeeze(0)

        if jacobian:
            return embedded, J
        else:
            return embedded

class VAE(nn.Module, EmbeddedManifold):
    def __init__(self, x, layers, num_components=100, device=None):
        super(VAE, self).__init__()

        self.device = device

        self.p = int(layers[0])  # Dimension of x
        self.d = int(layers[-1])  # Dimension of z
        self.h = layers  # [1:-1] # Dimension of hidden layers
        self.num_components = num_components

        enc = []
        for k in range(len(layers) - 1):
            in_features = int(layers[k])
            out_features = int(layers[k + 1])
            enc.append(nnj.ResidualBlock(nnj.Linear(in_features, out_features),
                                         nnj.Softplus()))
        enc.append(nnj.Linear(out_features, int(self.d * 2)))

        dec = []
        for k in reversed(range(len(layers) - 1)):
            in_features = int(layers[k + 1])
            out_features = int(layers[k])
            dec.append(nnj.ResidualBlock(nnj.Linear(in_features, out_features),
                                         nnj.Softplus()))
        dec.append(nnj.Sigmoid())

        # Note how we use 'nnj' instead of 'nn' -- this gives automatic
        # computation of Jacobians of the implemented neural network.
        # The embed function is required to also return Jacobians if
        # requested; by using 'nnj' this becomes a trivial constraint.
        self.encoder = nnj.Sequential(*enc)

        self.decoder_loc = nnj.Sequential(*dec)
        self.init_decoder_scale = 0.01 * torch.ones(self.p, device=self.device)

        self.prior_loc = torch.zeros(self.d, device=self.device)
        self.prior_scale = torch.ones(self.d, device=self.device)
        self.prior = td.Independent(td.Normal(loc=self.prior_loc, scale=self.prior_scale), 1)

        # Create a blank std-network.
        # It is important to call init_std after training the mean, but before training the std
        self.dec_std = None

        self.to(self.device)

    def encode(self, x):
        z_loc, z_scale = torch.chunk(self.encoder(x), chunks=2, dim=-1)
        return td.Independent(td.Normal(loc=z_loc, scale=z_scale.mul(0.5).exp() + 1e-10), 1)

    def decode(self, z):
        x_loc = self.decoder_loc(z)
        # return td.Independent(td.Normal(loc=x_loc, scale=self.decoder_scale(z) + 1e-10), 1)
        return td.Normal(loc=x_loc, scale=self.decoder_scale(z) + 1e-10)

    def decoder_scale(self, z, jacobian=False):
        if self.dec_std is None:
            return self.init_decoder_scale
        else:
            return self.dec_std(z, jacobian=jacobian)

    def elbo(self, x, kl_weight=1.0):
        q = self.encode(x)
        z = q.rsample()  # (batch size)x(latent dim)
        px_z = self.decode(z)  # p(x|z)
        ELBO = px_z.log_prob(x).mean() - kl_weight * KL(q, self.prior).mean()
        return ELBO

    def fit_mean(self, data_loader, num_epochs=150, num_cycles=30, max_kl=5):
        switch_epoch = num_epochs // 5
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        for cycle in range(1, num_cycles + 1):
            print('Cycle:', cycle)
            for epoch in range(num_epochs):
                sum_loss = 0
                # implements cyclic KL scaling
                if cycle < num_cycles - num_cycles // 5:
                    lam = min(max_kl, max(0.0, 2 * max_kl * (epoch - switch_epoch) / (num_epochs - switch_epoch)))
                else:
                    lam = 1
                for batch_idx, (data,) in enumerate(data_loader):
                    data = data.to(self.device)
                    optimizer.zero_grad()
                    loss = -self.elbo(data, lam)
                    loss.backward()
                    optimizer.step()
                    sum_loss += loss.item() * len(data)
                avg_loss = sum_loss / len(data_loader.dataset)
                print('(MEAN; lambda={:.4f}) ====> Epoch: {} Average loss: {:.4f}'.format(lam, epoch, avg_loss))

    # This function sets up the data structures for the RBF network for modeling variance.
    # XXX: We should do this more elagantly directly in __init__
    def init_std(self, x, gmm_mu=None, gmm_cv=None, weights=None):
        with torch.no_grad():
            z = torch.chunk(self.encoder(x.to(self.device)), chunks=2, dim=-1)[0]
        N, D = x.shape
        d = z.shape[1]
        inv_maxstd = 1e-1  # 1.0 / x.std(dim=0).mean() # x.std(dim=0).mean() #D*x.var(dim=0).mean()

        if gmm_mu is None and gmm_cv is None and weights is None:
            from sklearn import mixture
            clf = mixture.GaussianMixture(n_components=self.num_components, covariance_type='spherical')
            clf.fit(z.cpu().numpy())
            self.gmm_means = clf.means_
            self.gmm_covariances = clf.covariances_
            self.clf_weights = clf.weights_
        else:
            print('loading weights...')
            self.gmm_means = gmm_mu
            self.gmm_covariances = gmm_cv
            self.clf_weights = weights

        self.dec_std = nnj.Sequential(nnj.RBF(d, self.num_components,
                                              points=torch.tensor(self.gmm_means, dtype=torch.float,
                                                                  requires_grad=False),
                                              beta=0.5 / torch.tensor(self.gmm_covariances, dtype=torch.float,
                                                                      requires_grad=False)),  # d --> num_components
                                      nnj.PosLinear(self.num_components, 1, bias=False),  # num_components --> 1
                                      nnj.Reciprocal(inv_maxstd),  # 1 --> 1
                                      nnj.PosLinear(1, D)).to(self.device)  # 1 --> D
        with torch.no_grad():
            self.dec_std[1].weight[:] = ((torch.tensor(self.clf_weights, dtype=torch.float).exp() - 1.0).log()).to(
                self.device)
        self.dec_std

    def fit_std(self, data_loader, num_epochs):
        optimizer = torch.optim.Adam(self.dec_std.parameters(), weight_decay=1e-4)
        for epoch in range(num_epochs):
            sum_loss = 0
            for batch_idx, (data,) in enumerate(data_loader):
                data = data.to(self.device)
                optimizer.zero_grad()
                # XXX: we should sample here
                loss = -self.elbo(data, 1)
                loss.backward()
                sum_loss += loss.item() * len(data)
                optimizer.step()
            avg_loss = sum_loss / len(data_loader.dataset)
            print('(STD)  ====> Epoch: {} Average loss: {:.4f}'.format(epoch, avg_loss))

    def plot(self, z, labels, meshsize=150):
        import matplotlib.pyplot as plt
        if z.shape[1] == 2:
            minz, _ = z.min(dim=0)  # d
            maxz, _ = z.max(dim=0)  # d
            alpha = 0.1 * (maxz - minz)  # d
            minz -= alpha  # d
            maxz += alpha  # d

            ran0 = torch.linspace(minz[0].item(), maxz[0].item(), meshsize, device=self.device)
            ran1 = torch.linspace(minz[1].item(), maxz[1].item(), meshsize, device=self.device)
            Mx, My = torch.meshgrid(ran0, ran1)
            Mxy = torch.cat((Mx.t().reshape(-1, 1), My.t().reshape(-1, 1)), dim=1)  # (meshsize^2)x2
            with torch.no_grad():
                varim = self.decoder_scale(Mxy).pow(2).mean(dim=-1).reshape(meshsize, meshsize)
                varim = varim.cpu().detach().numpy()
            plt.imshow(varim, extent=(ran0[0].item(), ran0[-1].item(), ran1[0].item(), ran1[-1].item()), origin='lower')
            plt.colorbar()

            for label in labels.unique():
                idx = labels == label
                zi = z[idx].cpu().detach().numpy()
                plt.plot(zi[:, 0], zi[:, 1], '.')
        else:
            raise Exception('Latent dimension not suitable for plotting')

    def embed(self, points, jacobian=False):
        """
        Embed the manifold into (mu, std) space.

        Input:
            points:     a Nx(d) or BxNx(d) torch Tensor representing a (batch of a)
                        set of N points in latent space that will be embedded
                        in R^2D.

        Optional input:
            jacobian:   a boolean indicating if the Jacobian matrix of the function
                        should also be returned. Default is False.

        Output:
            embedded:   a Nx(2D) of BxNx(2D) torch tensor containing the N embedded points.
                        The first Nx(d) part contain the mean part of the embedding,
                        whlie the last Nx(d) part contain the standard deviation
                        embedding.

        Optional output:
            J:          If jacobian=True then a second Nx(2D)x(d) or BxNx(2D)x(d)
                        torch tensor is returned that contain the Jacobian matrix
                        of the embedding function.
        """
        std_scale = 1.0
        is_batched = points.dim() > 2
        if not is_batched:
            points = points.unsqueeze(0)  # BxNxD

        if jacobian:
            mu, Jmu = self.decoder_loc(points, jacobian=True)  # BxNxD, BxNxDx(d)
            std, Jstd = self.decoder_scale(points, jacobian=True)  # BxNxD, BxNxDx(d)
            embedded = torch.cat((mu, std_scale * std), dim=2)  # BxNx(2D)
            J = torch.cat((Jmu, std_scale * Jstd), dim=2)  # BxNx(2D)x(d)
        else:
            mu = self.decoder_loc(points)  # BxNxD
            std = self.decoder_scale(points)  # BxNxD
            embedded = torch.cat((mu, std_scale * std), dim=2)  # BxNx(2D)

        if not is_batched:
            embedded = embedded.squeeze(0)
            if jacobian:
                J = J.squeeze(0)

        if jacobian:
            return embedded, J
        else:
            return embedded


class linear_vae(nn.Module, EmbeddedManifold):
    def __init__(self):
        super(linear_vae, self).__init__()

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.encoder = nnj.Sequential(nnj.Linear(784, 256),
                                      nnj.Linear(256, 4))
        self.decoder_loc = nnj.Sequential(nnj.Linear(2, 256),
                                          nnj.Linear(256, 784))
        self.init_decoder_scale = 0.01 * torch.ones(784).to(self.device)

        self.prior_loc = torch.zeros(2).to(self.device)
        self.prior_scale = torch.ones(2).to(self.device)
        self.prior = td.Normal(loc=self.prior_loc, scale=self.prior_scale)

    def decoder_scale(self, z, jacobian=False):
        return self.init_decoder_scale

    def encode(self, x):
        z_loc, z_scale = torch.chunk(self.encoder(x), chunks=2, dim=-1)
        return td.Normal(loc=z_loc, scale=z_scale.mul(0.5).exp() + 1e-10)

    def decode(self, z):
        x_loc = self.decoder_loc(z)
        return td.Normal(loc=x_loc, scale=self.decoder_scale)

    def elbo(self, x, kl_weight=1.0):
        q = self.encode(x)
        z = q.rsample()  # (batch size)x(latent dim)
        px_z = self.decode(z)  # p(x|z)
        ELBO = px_z.log_prob(x).mean() - kl_weight * KL(q, self.prior).mean()
        return ELBO

    def fit_mean(self, data_loader, num_epochs=150, num_cycles=30, max_kl=5):
        switch_epoch = num_epochs // 6
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        for cycle in range(1, num_cycles + 1):
            print('Cycle:', cycle)
            for epoch in range(num_epochs):
                sum_loss = 0
                # implements cyclic KL scaling
                if cycle < num_cycles - num_cycles // 5:
                    lam = min(max_kl, max(0.0, 2 * max_kl * (epoch - switch_epoch) / (num_epochs - switch_epoch)))
                else:
                    lam = 1
                for batch_idx, (data,) in enumerate(data_loader):
                    data = data.to(self.device)
                    optimizer.zero_grad()
                    loss = -self.elbo(data, lam)
                    loss.backward()
                    optimizer.step()
                    sum_loss += loss.item() * len(data)
                avg_loss = sum_loss / len(data_loader.dataset)
                print('(MEAN; lambda={:.4f}) ====> Epoch: {} Average loss: {:.4f}'.format(lam, epoch, avg_loss))

    def plot(self, z, labels, meshsize=150):
        import matplotlib.pyplot as plt
        if z.shape[1] == 2:
            minz, _ = z.min(dim=0)  # d
            maxz, _ = z.max(dim=0)  # d
            alpha = 0.1 * (maxz - minz)  # d
            minz -= alpha  # d
            maxz += alpha  # d

            ran0 = torch.linspace(minz[0].item(), maxz[0].item(), meshsize, device=self.device)
            ran1 = torch.linspace(minz[1].item(), maxz[1].item(), meshsize, device=self.device)
            Mx, My = torch.meshgrid(ran0, ran1)
            Mxy = torch.cat((Mx.t().reshape(-1, 1), My.t().reshape(-1, 1)), dim=1)  # (meshsize^2)x2
            with torch.no_grad():
                varim = self.decoder_scale(Mxy).pow(2).mean(dim=-1).reshape(meshsize, meshsize)
                varim = varim.cpu().detach().numpy()
            plt.imshow(varim, extent=(ran0[0].item(), ran0[-1].item(), ran1[0].item(), ran1[-1].item()), origin='lower')
            plt.colorbar()

            for label in labels.unique():
                idx = labels == label
                zi = z[idx].cpu().detach().numpy()
                plt.plot(zi[:, 0], zi[:, 1], '.')
        else:
            raise Exception('Latent dimension not suitable for plotting')

    def embed(self, points, jacobian=False):
        """
        Embed the manifold into (mu, std) space.

        Input:
            points:     a Nx(d) or BxNx(d) torch Tensor representing a (batch of a)
                        set of N points in latent space that will be embedded
                        in R^2D.

        Optional input:
            jacobian:   a boolean indicating if the Jacobian matrix of the function
                        should also be returned. Default is False.

        Output:
            embedded:   a Nx(2D) of BxNx(2D) torch tensor containing the N embedded points.
                        The first Nx(d) part contain the mean part of the embedding,
                        whlie the last Nx(d) part contain the standard deviation
                        embedding.

        Optional output:
            J:          If jacobian=True then a second Nx(2D)x(d) or BxNx(2D)x(d)
                        torch tensor is returned that contain the Jacobian matrix
                        of the embedding function.
        """
        std_scale = 1.0
        is_batched = points.dim() > 2
        if not is_batched:
            points = points.unsqueeze(0)  # BxNxD

        if jacobian:
            mu, Jmu = self.decoder_loc(points, jacobian=True)  # BxNxD, BxNxDx(d)
            std = self.decoder_scale(points, jacobian=True).unsqueeze(0).unsqueeze(0).repeat(
                [mu.shape[0], mu.shape[1], 1])  # BxNxD, BxNxDx(d) JY: Changed because constant variance
            embedded = torch.cat((mu, std_scale * std), dim=2)  # BxNx(2D)
            # JY: Changed because constant variance
            J = Jmu  # BxNx(2D)x(d)
        else:
            mu = self.decoder_loc(points)  # BxNxD
            std = self.decoder_scale(points).unsqueeze(0).unsqueeze(0).repeat([mu.shape[0], mu.shape[1], 1])  # BxNxD
            embedded = torch.cat((mu, std_scale * std), dim=2)  # BxNx(2D)

        if not is_batched:
            embedded = embedded.squeeze(0)
            if jacobian:
                J = J.squeeze(0)

        if jacobian:
            return embedded, J
        else:
            return embedded


class linear_vae_decoder_variance(nn.Module, EmbeddedManifold):
    def __init__(self, num_components=50):
        super(linear_vae_decoder_variance, self).__init__()

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.encoder = nnj.Sequential(nnj.Linear(784, 256),
                                      nnj.Linear(256, 4))
        self.decoder_loc = nnj.Sequential(nnj.Linear(2, 256),
                                          nnj.Linear(256, 784))
        self.init_decoder_scale = 0.01 * torch.ones(784).to(self.device)
        self.dec_std = None
        self.num_components = num_components

        self.prior_loc = torch.zeros(2).to(self.device)
        self.prior_scale = torch.ones(2).to(self.device)
        self.prior = td.Normal(loc=self.prior_loc, scale=self.prior_scale)

    def decoder_scale(self, z, jacobian=False):
        if self.dec_std is None:
            return self.init_decoder_scale
        else:
            return self.dec_std(z, jacobian=jacobian)

    def encode(self, x):
        z_loc, z_scale = torch.chunk(self.encoder(x), chunks=2, dim=-1)
        return td.Normal(loc=z_loc, scale=z_scale.mul(0.5).exp() + 1e-10)

    def decode(self, z):
        x_loc = self.decoder_loc(z)
        return td.Normal(loc=x_loc, scale=self.decoder_scale(z))

    def elbo(self, x, kl_weight=1.0):
        q = self.encode(x)
        z = q.rsample()  # (batch size)x(latent dim)
        px_z = self.decode(z)  # p(x|z)
        ELBO = px_z.log_prob(x).mean() - kl_weight * KL(q, self.prior).mean()
        return ELBO

    def fit_mean(self, data_loader, num_epochs=150, num_cycles=30, max_kl=5):
        switch_epoch = num_epochs // 6
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        for cycle in range(1, num_cycles + 1):
            print('Cycle:', cycle)
            for epoch in range(num_epochs):
                sum_loss = 0
                # implements cyclic KL scaling
                if cycle < num_cycles - num_cycles // 5:
                    lam = min(max_kl, max(0.0, 2 * max_kl * (epoch - switch_epoch) / (num_epochs - switch_epoch)))
                else:
                    lam = 1
                for batch_idx, (data,) in enumerate(data_loader):
                    data = data.to(self.device)
                    optimizer.zero_grad()
                    loss = -self.elbo(data, lam)
                    loss.backward()
                    optimizer.step()
                    sum_loss += loss.item() * len(data)
                avg_loss = sum_loss / len(data_loader.dataset)
                print('(MEAN; lambda={:.4f}) ====> Epoch: {} Average loss: {:.4f}'.format(lam, epoch, avg_loss))

    def init_std(self, x, gmm_mu=None, gmm_cv=None, weights=None):
        with torch.no_grad():
            z = torch.chunk(self.encoder(x.to(self.device)), chunks=2, dim=-1)[0]
        N, D = x.shape
        d = z.shape[1]
        inv_maxstd = 1e-1  # 1.0 / x.std(dim=0).mean() # x.std(dim=0).mean() #D*x.var(dim=0).mean()

        if gmm_mu is None and gmm_cv is None and weights is None:
            from sklearn import mixture
            clf = mixture.GaussianMixture(n_components=self.num_components, covariance_type='spherical')
            clf.fit(z.cpu().numpy())
            self.gmm_means = clf.means_
            self.gmm_covariances = clf.covariances_
            self.clf_weights = clf.weights_
        else:
            print('loading weights...')
            self.gmm_means = gmm_mu
            self.gmm_covariances = gmm_cv
            self.clf_weights = weights

        self.dec_std = nnj.Sequential(nnj.RBF(d, self.num_components,
                                              points=torch.tensor(self.gmm_means, dtype=torch.float,
                                                                  requires_grad=False),
                                              beta=0.5 / torch.tensor(self.gmm_covariances, dtype=torch.float,
                                                                      requires_grad=False)),  # d --> num_components
                                      nnj.PosLinear(self.num_components, 1, bias=False),  # num_components --> 1
                                      nnj.Reciprocal(inv_maxstd),  # 1 --> 1
                                      nnj.PosLinear(1, D)).to(self.device)  # 1 --> D
        with torch.no_grad():
            self.dec_std[1].weight[:] = ((torch.tensor(self.clf_weights, dtype=torch.float).exp() - 1.0).log()).to(
                self.device)

    def fit_std(self, data_loader, num_epochs):
        optimizer = torch.optim.Adam(self.dec_std.parameters(), weight_decay=1e-4)
        for epoch in range(num_epochs):
            sum_loss = 0
            for batch_idx, (data,) in enumerate(data_loader):
                data = data.to(self.device)
                optimizer.zero_grad()
                # XXX: we should sample here
                loss = -self.elbo(data, 1)
                loss.backward()
                sum_loss += loss.item() * len(data)
                optimizer.step()
            avg_loss = sum_loss / len(data_loader.dataset)
            print('(STD)  ====> Epoch: {} Average loss: {:.4f}'.format(epoch, avg_loss))

    def plot(self, z, labels, meshsize=150):
        import matplotlib.pyplot as plt
        if z.shape[1] == 2:
            minz, _ = z.min(dim=0)  # d
            maxz, _ = z.max(dim=0)  # d
            alpha = 0.1 * (maxz - minz)  # d
            minz -= alpha  # d
            maxz += alpha  # d

            ran0 = torch.linspace(minz[0].item(), maxz[0].item(), meshsize, device=self.device)
            ran1 = torch.linspace(minz[1].item(), maxz[1].item(), meshsize, device=self.device)
            Mx, My = torch.meshgrid(ran0, ran1)
            Mxy = torch.cat((Mx.t().reshape(-1, 1), My.t().reshape(-1, 1)), dim=1)  # (meshsize^2)x2
            with torch.no_grad():
                varim = self.decoder_scale(Mxy).pow(2).mean(dim=-1).reshape(meshsize, meshsize)
                varim = varim.cpu().detach().numpy()
            plt.imshow(varim, extent=(ran0[0].item(), ran0[-1].item(), ran1[0].item(), ran1[-1].item()), origin='lower')
            plt.colorbar()

            for label in labels.unique():
                idx = labels == label
                zi = z[idx].cpu().detach().numpy()
                plt.plot(zi[:, 0], zi[:, 1], '.')
        else:
            raise Exception('Latent dimension not suitable for plotting')

    def embed(self, points, jacobian=False):
        """
        Embed the manifold into (mu, std) space.

        Input:
            points:     a Nx(d) or BxNx(d) torch Tensor representing a (batch of a)
                        set of N points in latent space that will be embedded
                        in R^2D.

        Optional input:
            jacobian:   a boolean indicating if the Jacobian matrix of the function
                        should also be returned. Default is False.

        Output:
            embedded:   a Nx(2D) of BxNx(2D) torch tensor containing the N embedded points.
                        The first Nx(d) part contain the mean part of the embedding,
                        whlie the last Nx(d) part contain the standard deviation
                        embedding.

        Optional output:
            J:          If jacobian=True then a second Nx(2D)x(d) or BxNx(2D)x(d)
                        torch tensor is returned that contain the Jacobian matrix
                        of the embedding function.
        """
        std_scale = 1.0
        is_batched = points.dim() > 2
        if not is_batched:
            points = points.unsqueeze(0)  # BxNxD

        if jacobian:
            mu, Jmu = self.decoder_loc(points, jacobian=True)  # BxNxD, BxNxDx(d)
            std, Jstd = self.decoder_scale(points, jacobian=True)  # BxNxD, BxNxDx(d)
            embedded = torch.cat((mu, std_scale * std), dim=2)  # BxNx(2D)
            J = torch.cat((Jmu, std_scale * Jstd), dim=2)  # BxNx(2D)x(d)
        else:
            mu = self.decoder_loc(points)  # BxNxD
            std = self.decoder_scale(points)  # BxNxD
            embedded = torch.cat((mu, std_scale * std), dim=2)  # BxNx(2D)

        if not is_batched:
            embedded = embedded.squeeze(0)
            if jacobian:
                J = J.squeeze(0)

        if jacobian:
            return embedded, J
        else:
            return embedded

