from typing import Optional, Tuple

import numpy as np


class AutoBackgroundSubtraction:
    DEFAULT_FITTING_DEGREE = 5
    DEFAULT_RESOLUTION = 1e-3
    DEFAULT_SEARCH_BEGIN = 0.
    DEFAULT_SEARCH_END = 1.

    def __init__(
            self, smooth_region: Tuple[int, int],
            fit_degree: Optional[int] = None,
            resolution: Optional[float] = None,
            search_begin: Optional[float] = None,
            search_end: Optional[float] = None) -> None:
        self.__smooth_region = smooth_region
        self.__fit_degree = fit_degree if fit_degree \
            else self.DEFAULT_FITTING_DEGREE
        self.__resolution = resolution if resolution \
            else self.DEFAULT_RESOLUTION
        self.__search_begin = search_begin if search_begin \
            else self.DEFAULT_SEARCH_BEGIN
        self.__search_end = search_end if search_end \
            else self.DEFAULT_SEARCH_END
        self.__coef_candidates: np.ndarray = np.arange(
            self.__search_begin, self.__search_end, self.__resolution)

    @property
    def coefficient_candidates(self) -> np.ndarray:
        return self.__coef_candidates

    def fitting_mse(self, raw: np.ndarray, bg: np.ndarray,
                    coef: float) -> float:
        """Returns error between subtracted spectrum and fitting curve of
        subtracted spectrum. Subtracted spectrum is
        computed as {raw - bg * coef}.

        Args:
            raw (np.ndarray): region to smooth from raw spectrum (1D)
                              containing background band(s)
            bg (np.ndarray): background spectrum (1D) corresponds to raw
            coef (float): coefficient for subtracting background

        Returns:
            float: error between subtracted spectrum and fitting curve
        """
        fit_x = np.arange(len(raw))
        subtracted = raw - bg * coef
        fit_coefs = np.polyfit(x=fit_x, y=subtracted, deg=self.__fit_degree)
        fit_curve = np.poly1d(fit_coefs)(fit_x)
        return np.average((subtracted - fit_curve) ** 2)

    def mse_array(self, raw: np.ndarray, bg: np.ndarray) -> np.ndarray:
        return np.array([self.fitting_mse(raw, bg, coef)
                         for coef in self.__coef_candidates])

    def crop_smooth_region(self, spectrum: np.ndarray) -> np.ndarray:
        return spectrum[self.__smooth_region[0]:self.__smooth_region[1]]

    def estimate_coef(self, raw_sp: np.ndarray, bg_sp: np.ndarray) -> float:
        raw_cropped = self.crop_smooth_region(raw_sp)
        bg_cropped = self.crop_smooth_region(bg_sp)

        mse_arr = self.mse_array(raw_cropped, bg_cropped)
        return self.__coef_candidates[np.argmin(mse_arr)]


class PhalanxRAutoBackgroundSubtraction(AutoBackgroundSubtraction):
    def __init__(
            self, smooth_region: Tuple[int, int],
            fit_degree: Optional[int] = None,
            resolution: Optional[float] = None,
            search_begin: Optional[float] = None,
            search_end: Optional[float] = None) -> None:
        super().__init__(smooth_region, fit_degree,
                         resolution, search_begin, search_end)

    def estimate_coef(self, raw_sp: np.ndarray, bg_sp: np.ndarray) -> float:
        raise NotImplementedError(
            "This class is only for 4-dimensional spectral image. "
            "Use AutoBackgroundSubtraction class for single spectrum "
            "or estimate_image_coef() method in this class "
            "for 4-dimensional image.")

    def get_average_spectrum(self, image: np.ndarray, x_region: tuple[int, int],
                             y_region: tuple[int, int]) -> np.ndarray:
        roi = image[x_region[0]:x_region[1]+1,
                    y_region[0]:y_region[1]+1, 0, :]
        return np.average(np.average(roi, axis=0), axis=0)

    def estimate_image_coef(self, raw_image: np.ndarray,
                            bg_image: np.ndarray,
                            x_region: Tuple[int, int],
                            y_region: Tuple[int, int]) -> float:
        raw_sp = self.get_average_spectrum(raw_image, x_region, y_region)
        bg_sp = self.get_average_spectrum(bg_image, x_region, y_region)
        return super().estimate_coef(raw_sp, bg_sp)
