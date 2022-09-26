from imports.general import *
from imports.ml import *
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import LeavePOut as LeaveKOut
from src.dataset import Dataset


class Recalibrator(object):
    def __init__(self, dataset: Dataset, model, mode: str = "cv", K: int = 1) -> None:
        self.K = K
        self.cv_module = LeaveKOut(self.K)
        self.mode = mode
        mus, sigmas, ys_true = self.make_recal_dataset(dataset, model)
        self.recalibrator_model = self.train_recalibrator_model(mus, sigmas, ys_true)

    def make_recal_dataset(self, dataset: Dataset, model):
        if self.mode == "cv":
            X_train, y_train = dataset.data.X_train, dataset.data.y_train
            mus, sigmas, ys_true = [], [], []
            for train_index, val_index in self.cv_module.split(X_train):
                X_train_, y_train_ = X_train[train_index, :], y_train[train_index]
                X_val_, y_val = X_train[val_index, :], y_train[val_index]
                model.fit(X_train_, y_train_)
                mus_val, sigs_val = model.predict(X_val_)
                if self.K > 1:
                    mus.extend(mus_val)
                    sigmas.extend(sigs_val)
                    ys_true.extend(y_val)
                else:
                    mus.append(mus_val)
                    sigmas.append(sigs_val)
                    ys_true.append(y_val.squeeze())
            return np.array(mus), np.array(sigmas), np.array(ys_true)
        elif self.mode == "iid":
            X_val_, y_val = dataset.data.X_test, dataset.data.y_test
            model.fit(dataset.data.X_train, dataset.data.y_train)
            mus_val, sigs_val = model.predict(X_val_)
            return mus_val, sigs_val, y_val

    def train_recalibrator_model(self, mu_test, sig_test, y_val):
        CDF = norm.cdf(y_val, mu_test, sig_test).squeeze()
        P = np.vectorize(lambda p: np.mean(CDF < p))
        P_hat = P(CDF)
        ir = IsotonicRegression(
            y_min=0, y_max=1, increasing=True, out_of_bounds="clip"
        ).fit(CDF, P_hat)
        calibrate_fun = lambda x: ir.predict(x)
        return calibrate_fun

    def estimate_moments_from_ecdf(self, y_space, cdf_hat):
        """
            Estimates the mean and variance from discritized CDF.
            Works by approximating the PDF by finite difference
        """
        pdf_hat = np.diff(cdf_hat)
        m1_hat = np.sum(y_space[1:] * pdf_hat)
        m2_hat = np.sum(y_space[1:] ** 2 * pdf_hat)
        v1_hat = m2_hat - m1_hat ** 2
        return m1_hat, v1_hat

    def recalibrate(self, mu_preds, sig_preds):
        is_tensor = torch.is_tensor(mu_preds)
        mu_preds = mu_preds.cpu().detach().numpy().squeeze() if is_tensor else mu_preds
        sig_preds = (
            sig_preds.cpu().detach().numpy().squeeze() if is_tensor else sig_preds
        )

        n_steps = 100
        mu_new, std_new = [], []
        for mu_i, std_i in zip(mu_preds, sig_preds):
            y_space = np.linspace(mu_i - 3 * std_i, mu_i + 3 * std_i, n_steps)
            cdf = norm.cdf(y_space, mu_i, std_i)
            cdf_hat = self.recalibrator_model(cdf)
            mu_i_hat, v_i_hat = self.estimate_moments_from_ecdf(y_space, cdf_hat)
            mu_new.append(mu_i_hat)
            std_new.append(np.sqrt(v_i_hat))

        if is_tensor:
            return (
                torch.from_numpy(np.array(mu_new)),
                torch.from_numpy(np.array(std_new)),
            )
        else:
            return np.array(mu_new), np.array(std_new)
