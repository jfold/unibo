from imports.general import *
from imports.ml import *


class SamplesExperiment(object):
    def __init__(self, big_plot: bool = True) -> None:
        self.n_calibration_bins = 20
        self.n_seeds = 100
        self.plot_it = big_plot
        self.mu_true, self.sigma_true = 0.0, 1.0
        self.n_calibration_bins = 25
        self.p = np.linspace(0, 1, self.n_calibration_bins)

        self.x_linspace = np.linspace(-5, 5, 100)
        self.pdf_true = norm.pdf(
            self.x_linspace, loc=self.mu_true, scale=self.sigma_true
        )
        self.cdf_true = norm.cdf(
            self.x_linspace, loc=self.mu_true, scale=self.sigma_true
        )

        self.sample_sizes = np.linspace(5, 100, 20).astype(np.int_)
        self.sample_sizes = np.append(self.sample_sizes, [200, 500, 1000, 1000])
        self.n_sample_sizes = len(self.sample_sizes)

        self.models = np.array(
            [[0, 1], [0, 2], [0, 0.1], [0.5, 1], [-1.5, 1], [0.5, 1.5]]
        )
        if not self.plot_it:
            n_models = 100
            model_mus = np.random.normal(0, 1, (n_models, 1))
            model_sigs = np.random.lognormal(1, 1, (n_models, 1))
            self.models = np.append(model_mus, model_sigs, axis=1)
        self.n_models = len(self.models)

    def fun2fit(self, x, alpha, beta, gamma):
        return beta * np.exp(-alpha * x) * (x ** gamma)

    def fit_fun(self, x, y):
        from scipy.optimize import curve_fit

        popt, _ = curve_fit(self.fun2fit, x, y)
        return popt

    def calibration(
        self, samples: np.ndarray, mus: np.ndarray, sigmas: np.ndarray
    ) -> np.ndarray:
        y_ = np.tile(samples, self.n_calibration_bins)
        p_array_ = np.tile(self.p[:, np.newaxis], sigmas.size)
        norms = tdist.Normal(
            torch.tensor(mus.squeeze()), torch.tensor(sigmas.squeeze())
        )
        icdfs = norms.icdf(torch.tensor(p_array_))
        calibrations = (
            torch.mean((torch.tensor(y_).T <= icdfs).float(), dim=1).cpu().numpy()
        )
        return calibrations

    def run(self):
        i_c = 1
        std_max = np.full((self.n_models, self.n_sample_sizes), np.nan)
        E_C = np.full((self.n_models, self.n_sample_sizes), np.nan)
        V_C = np.full((self.n_models, self.n_sample_sizes), np.nan)

        if self.plot_it:
            fig = plt.figure(figsize=(32, 12))

        p_matrix = np.tile(self.p[:, np.newaxis].T, (self.n_seeds, 1))
        for i_m, model in enumerate(self.models):
            if self.plot_it:
                ax = plt.subplot(self.n_models, 5, i_c)
                i_c += 1
                pdf = norm.pdf(self.x_linspace, loc=self.mu_true, scale=self.sigma_true)
                cdf = norm.cdf(self.x_linspace, loc=self.mu_true, scale=self.sigma_true)
                plt.plot(self.x_linspace, pdf, "-", color="black")
                plt.plot(self.x_linspace, cdf, "--", color="black")

            mu_model, sigma_model = model[0], model[1]

            if self.plot_it:
                pdf = norm.pdf(self.x_linspace, loc=mu_model, scale=sigma_model)
                cdf = norm.cdf(self.x_linspace, loc=mu_model, scale=sigma_model)
                if i_m == 0:
                    plt.title("Model")
                plt.plot(
                    self.x_linspace,
                    pdf,
                    "-",
                    color="blue",
                    label=r"$\mathcal{N}" + fr"({mu_model},{sigma_model}^2)$",
                )
                plt.plot(self.x_linspace, cdf, "--", color="blue")
                plt.legend(loc="upper left", prop={"size": 14})
                if i_m != self.n_models - 1:
                    ax.grid(True)
                    ax.set_xticklabels([])
                if i_m == self.n_models - 1:
                    plt.xlabel(r"$y_t$")

            data = np.full(
                (self.n_seeds, self.n_sample_sizes, self.n_calibration_bins), np.nan
            )

            for i_s, seed in enumerate(range(self.n_seeds)):
                np.random.seed(seed)
                for i_n, n_samples in enumerate(self.sample_sizes):
                    mus = np.tile(mu_model, (n_samples, 1))
                    sigmas = np.tile(sigma_model, (n_samples, 1))
                    samples = np.random.normal(
                        self.mu_true, self.sigma_true, (n_samples, 1)
                    )
                    data[i_s, i_n, :] = self.calibration(samples, mus, sigmas)

            for i_n, n_samples in enumerate(self.sample_sizes):
                mu = np.mean(data[:, i_n, :], axis=0)
                std = np.std(data[:, i_n, :], axis=0)
                std_max[i_m, i_n] = 2 * np.max(std)
                E_C[i_m, i_n] = np.mean((data[:, i_n, :] - p_matrix) ** 2)
                V_C[i_m, i_n] = np.std((data[:, i_n, :] - p_matrix) ** 2)
                if self.plot_it and n_samples in [10, 20, 50, 100]:
                    ax = plt.subplot(self.n_models, 5, i_c)
                    i_c += 1
                    if i_m == 0:
                        plt.title(fr"$N={n_samples}$")
                    plt.plot(self.p, mu, "-", color="blue")
                    E_C_str, V_C_str = "\mathbb{E}[E_C]", "\sqrt{\mathbb{V}[E_C]}"
                    plt.text(0, 0.75, fr"${E_C_str}= {E_C[i_m, i_n]:.3f}$")
                    plt.text(0.5, 0.0, fr"${V_C_str}= {V_C[i_m, i_n]:.3f}$")
                    plt.fill_between(
                        self.p, mu - 2 * std, mu + 2 * std, color="blue", alpha=0.2
                    )
                    plt.plot(self.p, self.p, "--", color="black")
                    if i_m != self.n_models - 1:
                        ax.grid(True)
                        ax.set_xticklabels([])
                    if i_m == self.n_models - 1:
                        ax.set_xlabel("Expected CI")
        if self.plot_it:
            fig.savefig("./figs/pdfs/calibration_nsamples.pdf")
            plt.show()

        # Supremum plot
        fig = plt.figure()
        mu = np.mean(std_max, axis=0)
        std = np.std(std_max, axis=0)
        # popt = self.fit_fun(self.sample_sizes, np.mean(std_max, axis=0))
        # fitted_fun = self.fun2fit(self.sample_sizes, popt[0], popt[1], popt[2])
        # print(f"{popt[0]:.2E},{popt[1]:.2f},{popt[2]:2f}")
        plt.plot(self.sample_sizes, mu, "-*", color="blue", label="Data")
        # plt.plot(self.sample_sizes, fitted_fun, "--", color="red", label="Fit")
        plt.plot(
            self.sample_sizes,
            np.sqrt(5) * mu[0] / np.sqrt(self.sample_sizes),
            "--",
            color="green",
            label=r"$a/\sqrt{n}$",
        )
        plt.legend()
        plt.fill_between(
            self.sample_sizes, mu - 2 * std, mu + 2 * std, color="blue", alpha=0.2
        )
        plt.ylabel(r"Max $2\sqrt{\mathbb{V} [C_y(p)]}$ ")
        plt.xlabel(r"$N$")
        plt.xscale("log")
        fig.savefig("./figs/pdfs/sup_std_calibration.pdf")
        # print(np.sqrt(5) * mu[0])
        # Expected Calibration error plot
        fig = plt.figure()
        mu = np.mean(E_C, axis=0)
        std = np.mean(V_C, axis=0) / np.sqrt(self.n_models)
        plt.plot(self.sample_sizes, mu, "-*", color="blue", label="Data")
        plt.fill_between(
            self.sample_sizes, mu - 2 * std, mu + 2 * std, color="blue", alpha=0.2
        )
        plt.legend()
        plt.ylabel(r"$\mathbb{E} [E_C]$ ")
        plt.xlabel(r"$N$")
        plt.xscale("log")
        fig.savefig("./figs/pdfs/E_C_calibration.pdf")

        # fig = plt.figure()
        # mu = np.mean(E_C, axis=0)
        # print(E_C.shape)
        # mu = [ee for ee in E_C]
        # plt.boxplot(mu)
        # # plt.fill_between(
        # #     self.sample_sizes, mu - 2 * std, mu + 2 * std, color="blue", alpha=0.2
        # # )
        # # plt.legend()
        # plt.ylabel(r"$\mathbb{E} [E_C]$ ")
        # plt.xlabel(r"$N$")
        # plt.xscale("log")
        # fig.savefig("./figs/pdfs/E_C_calibration.pdf")
