# put this file in cobaya/cobaya/likelihoods/lsst_y1

from cobaya.likelihood import Likelihood
import os
import sys

from cobaya.cobaya.theories.lsst_y1_cosmic_shear_emulator.cocoa_emu.config import cocoa_config

class lsst_emu_cs_lcdm(Likelihood):
    def initialize(self):
        super(lsst_emu_cs_lcdm,self)
        self.config = cocoa_config('./projects/lsst_y1/EXAMPLE_EVALUATE6.yaml')

    def get_requirements(self):
        return {
            'lsst_y1_xi': None
        }

    def add_baryon_q(self, Q, datavector):
        for i in range(self.n_pcas_baryon):
            datavector = datavector + Q[i] * self.baryon_pcas[:,i][0:self.dv_len]
        return datavector

    def logp(self, **params_values):
        model_datavector = self.provider.get_lsst_y1_xi()
        delta_dv = (model_datavector - self.config.dv_fid[0:780])[self.config.mask[0:780]]
        log_p = -0.5 * delta_dv @ self.config.cov_inv_masked @ delta_dv 
        return log_p












