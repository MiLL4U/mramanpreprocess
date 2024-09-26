import numpy as np

DEFAULT_THRESHOLD = 0.4
DEFAULT_PEAK_SEARCH_RANGE = 3
DEFAULT_FILL_WIDTH = 3
DEFAULT_FITTING_WIDTH = 5
DEFAULT_FITTING_ORDER = 2


class CosmicRayRemover:
    def __init__(
            self, threshold: float = DEFAULT_THRESHOLD,
            peak_search_range: int = DEFAULT_PEAK_SEARCH_RANGE,
            fill_width: int = DEFAULT_FILL_WIDTH,
            fitting_width: int = DEFAULT_FITTING_WIDTH,
            fitting_order: int = DEFAULT_FITTING_ORDER,
            print_cr_index: bool = False) -> None:
        self.__threshold = threshold
        self.__peak_search_range = peak_search_range
        self.__fill_width = fill_width
        self.__fitting_width = fitting_width
        self.__fitting_order = fitting_order
        self.__print_cr_index = print_cr_index

    def set_input(self, src_sp: np.ndarray) -> None:
        """set input spectrum to member variable"""
        self.__result = src_sp
        self.__sp_size = len(src_sp)

    def remove_cr(self, src_sp: np.ndarray) -> np.ndarray:
        """remove cosmic ray from input spectrum"""
        self.set_input(src_sp)
        while self.detect_cr():
            peak_loc = self.cr_peak_loc()
            if self.__print_cr_index:
                print(f"Found cosmic ray at: {peak_loc}")
            self.fill_value(peak_loc)
        return self.__result

    def minmax_normalize(self, src_sp: np.ndarray) -> np.ndarray:
        """normalize spectrum with minimum and max value"""
        max_ = np.max(src_sp)
        min_ = np.min(src_sp)
        return (src_sp - min_) / (max_ - min_)

    def update_normalized_gradient(self, src_sp: np.ndarray) -> None:
        """update normalized gradient"""
        normalized = self.minmax_normalize(src_sp)
        self.__norm_grad = np.diff(normalized, prepend=normalized[0])

    def detect_cr(self) -> bool:
        """detect cosmic ray and return whether spectrum has cosmic ray"""
        self.update_normalized_gradient(self.__result)
        return np.max(self.__norm_grad) > self.__threshold

    def grad_argmax(self) -> int:
        """return index where gradient is max"""
        return int(np.argmax(self.__norm_grad))

    def cr_peak_loc(self) -> int:
        """return index of a cosmic ray peak (most prominent)"""
        grad_argmax = self.grad_argmax()
        start_idx = max(0, grad_argmax - self.__peak_search_range)
        end_idx = min(self.__sp_size - 1, grad_argmax +
                      self.__peak_search_range)
        return start_idx + int(np.argmax(self.__result[start_idx:end_idx]))

    def fill_value(self, center_idx: int) -> None:
        """fill cosmic ray region with alternate values"""
        res = self.__result.copy()

        fill_x = self.fill_x(center_idx)
        fit_x = self.fit_x(center_idx)
        fit_coef = np.polyfit(x=fit_x, y=res[fit_x],
                              deg=self.__fitting_order)
        fit_y = np.poly1d(fit_coef)(fill_x)
        res[fill_x] = fit_y

        self.__result = res

    def fill_x(self, center_idx: int) -> np.ndarray:
        """region to fill with alternate values"""
        start = max(0, center_idx - self.__fill_width)
        end = min(self.__sp_size, center_idx + self.__fill_width + 1)
        return np.arange(start, end)

    def fit_x(self, center_idx: int) -> np.ndarray:
        """fitting region to get alternate values (neighbor of cosmic ray)"""
        fitting_mask = np.zeros_like(self.__result, dtype=bool)

        fill_x = self.fill_x(center_idx)
        fit_region = (center_idx - self.__fill_width - self.__fitting_width,
                      center_idx + self.__fill_width + self.__fitting_width + 1)

        fitting_mask[fit_region[0]:fit_region[1]] = True
        fitting_mask[fill_x] = False

        res = np.arange(len(self.__result))[fitting_mask]
        return res

    def fill_value_(self, center_idx: int) -> None:
        """fill specified region with average of neighboring values
        (deprecated)"""
        res = self.__result.copy()
        start_idx = max(0, center_idx - self.__fill_width)
        end_idx = min(self.__sp_size - 1, center_idx +
                      self.__fill_width)

        lower_start = start_idx - self.__fitting_width
        lower_end = end_idx - self.__fitting_width
        lower = res[lower_start:lower_end]

        higher_start = start_idx + self.__fitting_width
        higher_end = end_idx + self.__fitting_width
        higher = res[higher_start:higher_end]

        res[start_idx:end_idx] = ((lower + higher) / 2)
        self.__result = res


class MultiDimensionalCosmicRayRemover(CosmicRayRemover):
    def __init__(self, threshold: float = DEFAULT_THRESHOLD,
                 peak_search_range: int = DEFAULT_PEAK_SEARCH_RANGE,
                 fill_width: int = DEFAULT_FILL_WIDTH,
                 fitting_width: int = DEFAULT_FITTING_WIDTH,
                 fitting_order: int = DEFAULT_FITTING_ORDER,
                 print_cr_index: bool = False) -> None:
        super().__init__(threshold, peak_search_range, fill_width,
                         fitting_width, fitting_order, print_cr_index)

    def remove_cr(self, src_image: np.ndarray) -> np.ndarray:
        """remove cosmic ray from multi-dimensional spectral image"""
        spectra_2d = self.unpack_spectra(src_image)

        for i in range(len(spectra_2d)):
            spectra_2d[i, :] = super().remove_cr(spectra_2d[i])

        res = self.reconstruct_image(spectra_2d)
        return res

    def unpack_spectra(self, src_image: np.ndarray) -> np.ndarray:
        """extract spectra from multi-dimensional spectral image
        (get 2D array of spectra)
        """
        self.__img_shape = src_image.shape
        shape_2d = (np.prod(self.__img_shape[:-1]), self.__img_shape[-1])
        res = src_image.reshape(shape_2d)

        return res

    def reconstruct_image(self, spectra: np.ndarray) -> np.ndarray:
        """reconstruct multi-dimensional spectral image from array of spectra
        (reshape to original shape)"""
        return spectra.reshape(self.__img_shape)
