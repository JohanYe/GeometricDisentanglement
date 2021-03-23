import torch
import torch.nn as nn
from geoml import *
import geoml.nnj as nnj
from torch.nn.functional import softplus
import torch.distributions as td
from torch.distributions.kl import kl_divergence as KL
import matplotlib
import utils


class BasicVAE(nn.Module, EmbeddedManifold):
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

        dec = [nnj.ResidualBlock(nnj.Linear(2, hidden_layer[0]), nnj.Softplus())]
        for i in reversed(range(1, len(hidden_layer))):
            dec.append(nnj.ResidualBlock(nnj.Linear(hidden_layer[i], hidden_layer[i - 1]), nnj.Softplus()))
        dec.extend([nnj.ResidualBlock(nnj.Linear(hidden_layer[0], 784), nnj.Sigmoid())])
        self.decoder_loc = nnj.Sequential(*dec)
        #         self.decoder_loc = nnj.Sequential(nnj.ResidualBlock(nnj.Linear(hidden_layer[0], 784), nnj.Sigmoid()))

        self.init_decoder_scale = 0.01 * torch.ones(784, device=self.device)
        self.decoder_std = None

    def encode(self, x):
        z_mu, z_lv = torch.chunk(self.encoder(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=z_mu, scale=z_lv.mul(0.5).exp() + 1e-10), 1)

    def decode(self, z):
        x_mu = self.decoder_loc(z)
        x_std = self.decoder_scale(z)
        return td.Normal(loc=x_mu, scale=x_std + 1e-10)

    def decoder_scale(self, z):
        if self.decoder_std is None:
            return self.init_decoder_scale
        else:
            return self.decoder_std(z).mul(0.5).exp()

    def elbo(self, x, kl_weight=1):
        q = self.encode(x)
        z = q.rsample()  # (batch size)x(latent dim)
        px_z = self.decode(z)  # p(x|z)
        ELBO = px_z.log_prob(x).mean(-1) - kl_weight * KL(q, self.prior)
        return ELBO.mean()

    def fit_mean(self, data_loader, num_epochs=150, num_cycles=30, max_kl=5):
        switch_epoch = num_epochs // 3
        if self.decoder_std is not None:
            for layer in self.parameters():
                layer.requires_grad = True
            for layer in self.decoder_std.parameters():
                layer.requires_grad = False

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        train_loss = []

        for cycle in range(num_cycles):
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
                train_loss.append(avg_loss)
                print('(MEAN; lambda={:.4f}) ====> Epoch: {} Average loss: {:.4f}'.format(lam, epoch, avg_loss))

        return train_loss

    def init_std_naive(self):
        dec = [nnj.Linear(self.latent_space, self.hidden_layer[-1]), nnj.Softplus()]
        for i in reversed(range(1, len(self.hidden_layer))):
            dec.append(nnj.ResidualBlock(nnj.Linear(self.hidden_layer[i], self.hidden_layer[i - 1]), nnj.Softplus()))
        dec.extend([nnj.Linear(self.hidden_layer[0], 784)])
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

    def IWAE_eval(self, data_loader, k_samples, sum=False):
        """ Lazy method of IWAE eval """
        k = torch.Tensor([k_samples]).to(self.device)
        with torch.no_grad():
            sum_loss, n_samples = 0, 0
            for batch_idx, (data,) in enumerate(data_loader):
                log_px_z = None
                data = data.to(self.device)
                for i in range(int(k.item())):
                    q = self.encode(data)
                    z = q.rsample()  # (batch size)x(latent dim)
                    px_z = self.decode(z)  # p(x|z)

                    if log_px_z is None:
                        log_px_z = px_z.log_prob(data).sum(-1).unsqueeze(1) if sum else px_z.log_prob(data).mean(
                            -1).unsqueeze(1)
                        kld = KL(q, self.prior).unsqueeze(1)
                    else:
                        if sum:
                            log_px_z = torch.cat([log_px_z, px_z.log_prob(data).sum(-1).unsqueeze(1)], 1)
                        else:
                            log_px_z = torch.cat([log_px_z, px_z.log_prob(data).mean(-1).unsqueeze(1)], 1)
                        kld = torch.cat([kld, KL(q, self.prior).unsqueeze(1)], 1)

                log_wk = log_px_z - kld
                if log_wk.sum() > 0:
                    log_wk = -1 * log_wk
                L_k = log_wk.logsumexp(dim=-1) - k.log()

                loss = L_k.sum()
                sum_loss += loss.item()
            avg_loss = -1 * (sum_loss / len(data_loader.dataset))
            print('(TEST)  ====> Average loss: {:.4f}'.format(avg_loss))
        return avg_loss

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
        for k in range(len(layers) - 2):
            in_features = int(layers[k])
            out_features = int(layers[k + 1])
            enc.append(nnj.ResidualBlock(nnj.Linear(in_features, out_features),
                                         nnj.Softplus()))
        enc.append(nnj.Linear(out_features, int(self.d * 2)))

        dec = []
        for k in reversed(range(len(layers) - 1)):
            in_features = int(layers[k + 1])
            out_features = int(layers[k])
            if out_features != layers[0]:
                dec.append(nnj.ResidualBlock(nnj.Linear(in_features, out_features),
                                             nnj.Softplus()))
            else:
                dec.append(nnj.ResidualBlock(nnj.Linear(in_features, out_features),
                                             nnj.Sigmoid()))

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
        return td.Normal(loc=x_loc, scale=self.decoder_scale(z) + 1e-15)

    def decoder_scale(self, z, jacobian=False):
        if self.dec_std is None:
            return self.init_decoder_scale
        else:
            return self.dec_std(z, jacobian=jacobian)

    def elbo(self, x, kl_weight=1.0):
        q = self.encode(x)
        z = q.rsample()  # (batch size)x(latent dim)
        px_z = self.decode(z)  # p(x|z)
        ELBO = px_z.log_prob(x).mean(-1) - kl_weight * KL(q, self.prior)
        return ELBO.mean()

    def fit_mean(self, data_loader, num_epochs=150, num_cycles=30, max_kl=5):
        switch_epoch = num_epochs // 5
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

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
    def init_std(self, x, gmm_mu=None, gmm_cv=None, weights=None, beta_constant=0.5, beta_override=None,
                 inv_maxstd=7.5e-2, n_samples=2, num_components=None):
        self.beta_constant = beta_constant
        if num_components is not None:
            self.num_components = num_components
        N, D = x.shape
        with torch.no_grad():
            z = self.encode(x.to(self.device)).sample([n_samples]).reshape(n_samples*N, 2)
        d = z.shape[1]
        inv_maxstd = inv_maxstd  # 1.0 / x.std(dim=0).mean() # x.std(dim=0).mean() #D*x.var(dim=0).mean()

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
        if beta_override is None:
            self.beta = beta_constant / torch.tensor(self.gmm_covariances, dtype=torch.float, requires_grad=False)
        else:
            self.beta = beta_override
        self.beta = self.beta.to(self.device)
        self.dec_std = nnj.Sequential(nnj.RBF(d, self.num_components,
                                              points=torch.tensor(self.gmm_means, dtype=torch.float,
                                                                  requires_grad=False),
                                              beta=self.beta),  # d --> num_components
                                      nnj.PosLinear(self.num_components, 1, bias=False),  # num_components --> 1
                                      nnj.Reciprocal(inv_maxstd),  # 1 --> 1
                                      nnj.PosLinear(1, D)).to(self.device)  # 1 --> D
        with torch.no_grad():
            self.dec_std[1].weight[:] = ((torch.tensor(self.clf_weights, dtype=torch.float).exp() - 1.0).log()).to(
                self.device)
        self.dec_std

    def fit_std(self, train_loader, test_loader, num_epochs):
        optimizer = torch.optim.Adam(self.dec_std.parameters(), weight_decay=1e-4)
        for epoch in range(num_epochs):
            sum_loss = 0
            for batch_idx, (data,) in enumerate(train_loader):
                data = data.to(self.device)
                optimizer.zero_grad()
                # XXX: we should sample here
                loss = -self.elbo(data, 1)
                loss.backward()
                sum_loss += loss.item() * len(data)
                optimizer.step()
            avg_train_loss = sum_loss / len(train_loader.dataset)
            sum_loss = 0
            with torch.no_grad():
                for batch_idx, (data,) in enumerate(test_loader):
                    data = data.to(self.device)
                    loss = -self.elbo(data, 1)
                    sum_loss += loss.item() * len(data)
                avg_test_loss = sum_loss / len(test_loader.dataset)
            print('(STD)  ====> Epoch: {} Average train loss: {:.4f} Average test loss: {:.4f}'.format(epoch,
                                                                                                       avg_train_loss,
                                                                                                       avg_test_loss))

    def IWAE_eval(self, data_loader, k_samples, sum=False):
        """ Lazy method of IWAE eval """
        k = torch.Tensor([k_samples]).to(self.device)
        with torch.no_grad():
            sum_loss, n_samples = 0, 0
            for batch_idx, (data,) in enumerate(data_loader):
                log_px_z = None
                data = data.to(self.device)
                for i in range(int(k.item())):
                    q = self.encode(data)
                    z = q.rsample()  # (batch size)x(latent dim)
                    px_z = self.decode(z)  # p(x|z)

                    if log_px_z is None:
                        log_px_z = px_z.log_prob(data).sum(-1).unsqueeze(1) if sum else px_z.log_prob(data).mean(
                            -1).unsqueeze(1)
                        kld = KL(q, self.prior).unsqueeze(1)
                    else:
                        if sum:
                            log_px_z = torch.cat([log_px_z, px_z.log_prob(data).sum(-1).unsqueeze(1)], 1)
                        else:
                            log_px_z = torch.cat([log_px_z, px_z.log_prob(data).mean(-1).unsqueeze(1)], 1)
                        kld = torch.cat([kld, KL(q, self.prior).unsqueeze(1)], 1)

                log_wk = log_px_z - kld
                if log_wk.sum() > 0:
                    log_wk = -1 * log_wk
                L_k = log_wk.logsumexp(dim=-1) - k.log()

                loss = -L_k.sum()
                sum_loss += loss.item()
            avg_loss = -1 * (sum_loss / len(data_loader.dataset))
            print('(TEST)  ====> Average loss: {:.4f}'.format(avg_loss))
        return avg_loss

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


class VAE_bodies(nn.Module, EmbeddedManifold):
    def __init__(self, x, layers, num_components=100, device=None, old=False):
        super(VAE_bodies, self).__init__()

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
            if not old:  # temporary to load old models TODO: delete
                if out_features != layers[0]:
                    dec.append(nnj.ResidualBlock(nnj.Linear(in_features, out_features),
                                                 nnj.Softplus()))
                else:
                    dec.append(nnj.ResidualBlock(nnj.Linear(in_features, out_features),
                                                 nnj.Sigmoid()))
            else:
                dec.append(nnj.ResidualBlock(nnj.Linear(in_features, out_features),
                                             nnj.Softplus()))
                if out_features == layers[0]:
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
    def init_std(self, x, gmm_mu=None, gmm_cv=None, weights=None, inv_maxstd=1e-1, beta_constant=0.5,
                 component_overwrite=None, beta_override=None, n_samples=2, z_override=None, sigma=None):
        if component_overwrite is not None:
            self.num_components = component_overwrite
        if z_override is None:
            with torch.no_grad():
                mu, lv = torch.chunk(self.encoder(x.to(self.device)), chunks=2, dim=-1)
                z = td.Normal(loc=mu, scale=lv.mul(0.5).exp() + 1e-10).sample([n_samples])
                z = z.reshape(int(x.shape[0] * n_samples), z.shape[-1])
        else:
            z = z_override
        N, D = x.shape
        d = z.shape[1]
        inv_maxstd = inv_maxstd  # 1.0 / x.std(dim=0).mean() # x.std(dim=0).mean() #D*x.var(dim=0).mean()

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
        if beta_override is None:
            beta = beta_constant.cpu() / torch.tensor(self.gmm_covariances, dtype=torch.float, requires_grad=False)
        else:
            beta = beta_override
        self.beta = beta.to(self.device)
        self.dec_std = nnj.Sequential(nnj.RBF(d, self.num_components,
                                              points=torch.tensor(self.gmm_means, dtype=torch.float,
                                                                  requires_grad=False),
                                              beta=self.beta),  # d --> num_components
                                      nnj.PosLinear(self.num_components, 1, bias=False),  # num_components --> 1
                                      nnj.Reciprocal(inv_maxstd),  # 1 --> 1
                                      nnj.PosLinear(1, D)).to(self.device)  # 1 --> D
        if sigma is not None:
            self.dec_std[0] = nnj.RBF_variant(d, self.gmm_means.shape[0],
                                              points=torch.tensor(self.gmm_means, dtype=torch.float,
                                                                  requires_grad=False),
                                              beta=self.beta.requires_grad_(False), boxwidth=sigma).to(self.device)
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
