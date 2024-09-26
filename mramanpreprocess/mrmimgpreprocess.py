from typing import Tuple, Union

import ibwpy as ip
import numpy as np

from .crremover import MultiDimensionalCosmicRayRemover

DEFAULT_RESULT_PREFIX = "p"
DEFAULT_RESULT_SUFFIX = ""
INTENSITY_REFERENCE_REGION = (424, 867)  # integration range (O-H str.)


class MRamanImagePreprocess:
    def __init__(self, src_ibw: ip.BinaryWave5):
        self.__src_ibw = src_ibw
        self.__array = src_ibw.array
        self.__sp_scale = self.__src_ibw.axis_scale(self.ndim - 1)

    @property
    def source_name(self) -> str:
        return self.__src_ibw.name

    @property
    def array(self) -> np.ndarray:
        return self.__array

    @array.setter
    def array(self, new_array: np.ndarray) -> None:
        self.__array = new_array

    @property
    def ndim(self) -> int:
        return self.__array.ndim

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.__array.shape

    @property
    def spectra_num(self) -> np.int64:
        return np.prod(self.__array.shape[:-1])

    @property
    def channel_size(self) -> int:
        return self.__array.shape[-1]

    @property
    def all_spectra(self) -> np.ndarray:
        return np.reshape(self.__array, (self.spectra_num, self.channel_size))

    def get_ibw(self, name: Union[str, None] = None,
                duplicate_note: bool = True,
                duplicate_axis_scale: bool = True,
                dtype: type = np.float32) -> ip.BinaryWave5:
        res = ip.from_nparray(self.__array.astype(dtype), name)
        if duplicate_note:
            res.set_note(self.__src_ibw.note)
        if duplicate_axis_scale:
            for dim_idx in range(self.ndim - 1):  # spatial axis
                if self.__array.shape[dim_idx] == self.__src_ibw.shape[dim_idx]:
                    src_scale = self.__src_ibw.axis_scale(dim_idx)
                    res.set_axis_scale(dim_idx, src_scale[0], src_scale[1])
            res.set_axis_scale(  # spectral axis
                self.ndim - 1, self.__sp_scale[0], self.__sp_scale[1])
        return res

    def save_ibw(self, name: Union[str, None] = None,
                 path: Union[str, None] = None,
                 duplicate_note: bool = True,
                 duplicate_axis_scale: bool = True,
                 dtype: type = np.float32) -> None:
        if name is None:
            name = DEFAULT_RESULT_PREFIX + self.source_name + \
                DEFAULT_RESULT_SUFFIX
        if path is None:
            path = name + ".ibw"

        res = self.get_ibw(name, duplicate_note, duplicate_axis_scale, dtype)
        res.save(path)

    def subtract_bg(self, bg_ibw: ip.BinaryWave5, ratio: float) -> None:
        self.__array = np.array(self.__array - bg_ibw.array * ratio)

    def normalize_with_ref_integration(
            self, ref_ibw: ip.BinaryWave5,
            ref_region: Tuple[int, int] = INTENSITY_REFERENCE_REGION) -> None:
        integrations = np.sum(
            ref_ibw.array[:, :, 0, ref_region[0]:ref_region[1]], axis=2)
        self.__array = self.__array / integrations[:, :, None, None]

    def svd_noise_reduction(self, component_num: int,
                            print_name: bool = True) -> None:
        if print_name:
            print(f"Applying noise reduction on {self.source_name}...")
        U, s, V = np.linalg.svd(self.all_spectra, full_matrices=True)
        s_selected = np.zeros((U.shape[1], V.shape[0]))
        for i in range(component_num):
            s_selected[i][i] = s[i]
        reconstructed = np.dot(np.dot(U, s_selected), V)
        res = np.reshape(reconstructed, self.shape)
        self.__array = res

    def remove_cosmic_ray(self) -> None:
        remover = MultiDimensionalCosmicRayRemover(print_cr_index=True)
        self.__array = remover.remove_cr(self.__array)

    def spatial_binning(self, binning_shape: Tuple[int, int],
                        keep_shape: bool = False) -> None:
        if self.ndim != 4:
            raise ValueError("invalid number of dimensions")
        if self.shape[2] != 1:
            raise ValueError("only x-y spectral image is supported")

        orig_spatial_shape = self.shape[0:2]
        if not all([orig_size % binning_size == 0
                    for orig_size, binning_size
                    in zip(orig_spatial_shape, binning_shape)]):
            raise ValueError(
                'original spatial shape is not divisible by binning_shape')

        orig_3d_array = self.array.reshape(
            orig_spatial_shape + (self.channel_size,))

        x_split_pos = np.arange(0, orig_spatial_shape[0], binning_shape[0])[1:]
        y_split_pos = np.arange(0, orig_spatial_shape[1], binning_shape[1])[1:]

        y_blocks = np.split(orig_3d_array, x_split_pos, axis=0)
        blocks = np.array([np.split(y_block, y_split_pos, axis=1)
                           for y_block in y_blocks])
        res = np.array(np.average(np.average(blocks, axis=2), axis=2))

        res_shape = res.shape[0:2] + (1, res.shape[2])
        res = res.reshape(res_shape)

        if keep_shape:
            res = extend_shape(res, binning_shape)

        self.__array = res

    def trim_spectra(self, start: int = 0,
                     end: Union[int, None] = None) -> None:
        if end is None:
            end = self.shape[-1]
        new_channel_size = end - start
        res_spectra = self.all_spectra[:, start:end]
        res = res_spectra.reshape(self.shape[:-1] + (new_channel_size,))

        self.__array = res  # overwrite array

        # update start of spectral axis
        new_sp_start = self.__sp_scale[0] + self.__sp_scale[1] * start
        self.__sp_scale = (new_sp_start, self.__sp_scale[1])

    def correct_baseline(self, x_wave: np.ndarray,
                         mask: np.ndarray, order: int = 2) -> None:
        print(f"applying baseline correction ({self.source_name})...")
        spectra = self.all_spectra
        x_masked = x_wave[mask]
        res = np.empty(spectra.shape)
        for spectra_num in range(len(spectra)):
            raw = spectra[spectra_num]
            bl_obs = raw[mask]
            coefs = np.polyfit(x_masked, bl_obs, order)

            fit_bl = np.poly1d(coefs)(x_wave)
            bl_corrected = raw - fit_bl

            res[spectra_num] = bl_corrected
        self.__array = np.reshape(res, self.shape)

    def box_smooth(self, smooth_size: int) -> None:
        print(f"applying box smoothing ({self.source_name})...")
        smooth_filter = np.ones(smooth_size) / smooth_size

        spectra = self.all_spectra
        res = np.empty(spectra.shape)
        for spectrum_idx in range(len(spectra)):
            smoothed = np.convolve(
                spectra[spectrum_idx], smooth_filter, mode="same")
            res[spectrum_idx] = smoothed
        self.__array = np.reshape(res, self.shape)

    def normalize_with_integrate(self, region: Tuple[int, int]) -> None:
        print(f"applying normalization with integrate ({self.source_name})...")
        spectra = self.all_spectra

        integrates = np.sum(spectra[:, region[0]:region[1]], axis=1)
        divisor = np.array([integrates]).T
        res = spectra / divisor

        self.__array = np.reshape(res, self.shape)

    def divide_by_array(self, divisor: np.ndarray) -> None:
        if self.__array.shape != divisor.shape:
            raise ValueError("invalid shape of divisor array")
        res = self.__array / divisor
        self.__array = res


# def spatial_binning(
#         src_ibw: ip.BinaryWave5,
#         binning_shape: Tuple[int, int],
#         keep_shape: bool = False) -> ip.BinaryWave5:
#     if src_ibw.ndim != 4:
#         raise ValueError("invalid number of dimensions")
#     if src_ibw.shape[2] != 1:
#         raise ValueError("only x-y spectral image is supported")

#     orig_spatial_shape = src_ibw.shape[0:2]
#     if not all([orig_size % binning_size == 0
#                 for orig_size, binning_size
#                 in zip(orig_spatial_shape, binning_shape)]):
#         raise ValueError(
#             'original spatial shape is not divisible by binning_shape')

#     orig_3d_array = src_ibw.array.reshape(
#         orig_spatial_shape + (src_ibw.shape[-1],))

#     x_split_pos = np.arange(0, orig_spatial_shape[0], binning_shape[0])[1:]
#     y_split_pos = np.arange(0, orig_spatial_shape[1], binning_shape[1])[1:]

#     y_blocks = np.split(orig_3d_array, x_split_pos, axis=0)
#     blocks = np.array([np.split(y_block, y_split_pos, axis=1)
#                        for y_block in y_blocks])
#     res_arr = np.array(np.average(np.average(blocks, axis=2), axis=2))

#     res_shape = res_arr.shape[0:2] + (1, res_arr.shape[2])
#     res_arr = res_arr.reshape(res_shape)

#     if keep_shape:
#         res_arr = extend_shape(res_arr, binning_shape)

#     res = ip.from_nparray(res_arr, src_ibw.name)
#     return res


def extend_shape(src_image: np.ndarray,
                 extend_size: Tuple[int, int]) -> np.ndarray:
    res_shape = (src_image.shape[0] * extend_size[0], src_image.shape[1]
                 * extend_size[1]) + src_image.shape[2:]
    res = np.zeros(res_shape)
    for row in range(extend_size[0]):
        for col in range(extend_size[1]):
            res[row::extend_size[0], col::extend_size[1]] = src_image
    return res
