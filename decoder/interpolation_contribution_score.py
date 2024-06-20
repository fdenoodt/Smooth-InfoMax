from typing import Tuple, Dict

import numpy as np
import torch
import IPython.display as ipd

from config_code.config_classes import OptionsConfig
from decoder.eval_decoder import _get_models, _reconstruct_audio, _get_data, _get_all_data
from decoder.lit_decoder import LitDecoder


class InterpolationContributionScore:
    def __init__(self, opt: OptionsConfig, nb_dims: int, lit_decoder: LitDecoder):
        self.opt = opt
        self.nb_dims = nb_dims
        self.latent_nb_frames = (
            opt.decoder_config.retrieve_correct_decoder_architecture()).expected_nb_frames_latent_repr
        self.lit_decoder = lit_decoder

    def _get_two_zs(self, z, filenames, idx1, idx2, print_names: bool = True) -> (np.ndarray, np.ndarray, str, str):
        assert idx1 != idx2

        z = z.cpu().numpy()
        z1 = z[idx1].reshape(1, self.nb_dims, -1)  # (1, 512, 64)
        z2 = z[idx2].reshape(1, self.nb_dims, -1)  # (1, 512, 64)
        z1_file = filenames[idx1]
        z2_file = filenames[idx2]
        if print_names:
            print(z1_file, z2_file)

        if idx1 in [27, 15] and idx2 in [27, 15]:
            assert z1_file in ["bibibi_1", "bagaga_1"]
            assert z2_file in ["bibibi_1", "bagaga_1"]

        return z1, z2, z1_file, z2_file

    def _interpolate(self, z1, z2, nb_interpolations=10):
        vals = np.linspace(0, 1, nb_interpolations)
        z_interpolated = np.stack([z1 * val + z2 * (1 - val) for val in vals])
        return z_interpolated

    def _interpolate_partial(self, z1, z2, indices, nb_interpolations=10):
        z_interpolated = np.zeros((nb_interpolations, self.nb_dims, self.latent_nb_frames))
        for i in range(nb_interpolations):
            val = i / (nb_interpolations - 1)
            z_interpolated[i] = z1.copy()
            z_interpolated[i, indices] = z1[0, indices] * val + z2[0, indices] * (1 - val)
        # return z_interpolated

        # to tensor
        z_interpolated = torch.from_numpy(z_interpolated).float().to(self.opt.device)
        return z_interpolated

    def _dist_after_interpol_important_dims(self, z1, z2, nb_dims_important_dims, max_err: bool = False):
        assert nb_dims_important_dims <= self.nb_dims

        z_1_single_timeframe = z1.mean(axis=2)  # (1, 32)
        z_2_single_timeframe = z2.mean(axis=2)  # (1, 32)

        # mse
        mse = (z_1_single_timeframe - z_2_single_timeframe) ** 2
        dims = np.argsort(mse.squeeze())[::-1][:nb_dims_important_dims]  # in descending order

        if max_err:
            z2_partial = self._interpolate_partial(z1, z2, dims, 10)[-1]  # take the least interpolated z
        else:
            z2_partial = self._interpolate_partial(z1, z2, dims, 10)[
                0]  # take the most interpolated z (same as having a mask)

        x2_partial = self.lit_decoder.decoder(z2_partial).squeeze()
        z2 = torch.from_numpy(z2).float().to(self.opt.device)  # to tensor
        x2_target = self.lit_decoder.decoder(z2).squeeze()

        # dist = np.abs(x2_partial - x2_target)
        # dist = dist.mean()

        # MSE
        dist = torch.mean((x2_partial - x2_target) ** 2).item()

        # assert dist is a scalar, not a tensor
        assert np.isscalar(dist)

        return dist

    def compute_score(self) -> Dict[int, float]:
        self.lit_decoder = self.lit_decoder.to(self.opt.device)
        self.lit_decoder.eval()

        _, _, z, filenames = _get_all_data(self.opt, self.lit_decoder)
        nb_files = len(filenames)

        # calc max error
        max_error = 0
        for i in range(nb_files):
            for j in range(nb_files):
                if i != j:
                    z1, z2, z1_file, z2_file = self._get_two_zs(z, filenames, idx1=i, idx2=j, print_names=False)
                    dist = self._dist_after_interpol_important_dims(z1, z2, 512, max_err=True)
                    max_error += dist

                if j == 10:  # TODO: remove this
                    break
            if i == 10:
                break

        max_error /= (nb_files * (nb_files - 1))
        print(f"Max error: {max_error}")

        results: Dict[int, float] = {}
        for nb_most_important_dims in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
            avg_error = 0
            print(f"nb_most_important_dims: {nb_most_important_dims}")
            for i in range(nb_files):
                for j in range(nb_files):
                    if i != j:
                        z1, z2, z1_file, z2_file = self._get_two_zs(z, filenames, idx1=i, idx2=j, print_names=False)
                        dist = self._dist_after_interpol_important_dims(z1, z2, nb_most_important_dims)
                        avg_error += dist

                    if j == 10:  # TODO: remove this
                        break
                if i == 10:
                    break

            avg_error /= (nb_files * (nb_files - 1))
            # print(f"Unscaled average error for dim {nb_most_important_dims}: {avg_error}")
            # print(f"Scaled average error for dim {nb_most_important_dims}: {avg_error / max_error}")
            normalized_avg_error = avg_error / max_error
            results[nb_most_important_dims] = normalized_avg_error

        return results
