import astropy.units as u
import ndcube
import numpy as np
import sunpy.map

from sunkit_dem import GenericModel
from demregpy.dn2dem import dn2dem
from synthesizAR.instruments.util import extend_celestial_wcs


__all__ = ['HK12Model']


class HK12Model(GenericModel):
    
    def _model(self, alpha=1.0, increase_alpha=1.5, max_iterations=10, guess=None, use_em_loci=False, **kwargs):
        errors = np.array([self.data[k].uncertainty.array.squeeze() for k in self._keys]).T
        dem, edem, elogt, chisq, dn_reg = dn2dem(
            self.data_matrix.T,
            errors,
            self.kernel_matrix.T,
            np.log10(self.kernel_temperatures.to_value('K')),
            self.temperature_bin_edges.to_value('K'),
            max_iter=max_iterations,
            reg_tweak=alpha,
            rgt_fact=increase_alpha,
            dem_norm0=guess,
            gloci=use_em_loci,
            **kwargs,
        )
        _key = self._keys[0]
        dem_unit = self.data[_key].unit / self.kernel[_key].unit / self.temperature_bin_edges.unit
        uncertainty = edem.T * dem_unit
        em = (dem * np.diff(self.temperature_bin_edges)).T * dem_unit
        dem = dem.T * dem_unit
        T_error_upper = self.temperature_bin_centers * (10**elogt - 1 )
        T_error_lower = self.temperature_bin_centers * (1 - 1 / 10**elogt)
        return {'dem': dem,
                'uncertainty': uncertainty,
                'em': em,
                'temperature_errors_upper': T_error_upper.T,
                'temperature_errors_lower': T_error_lower.T,
                'chi_squared': np.atleast_1d(chisq).T}

    @classmethod
    def defines_model_for(self, *args, **kwargs):
        return kwargs.get('model') == 'hk12'
