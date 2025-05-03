import numpy as np
import os
import sys
import torch
import torch.nn as nn
from cobaya.theory import Theory
from cobaya.typing import InfoDict
import h5py as h5

sys.path.append(os.path.dirname(__file__))
from cocoa_emu.nn_emulator import \
    Better_Attention, Better_Transformer, Better_ResBlock, Affine


class lsst_y1_cosmic_shear_emulator(Theory):
    extra_args: InfoDict = { }

    def initialize(self):
        super().initialize()

        PATH = os.environ.get("ROOTDIR") + "/" + self.extra_args.get('filename')

        layers = []
        in_dim=12
        int_dim_res = 256
        n_channels = 32
        int_dim_trf = 1024
        out_dim = 780
        layers.append(nn.Linear(in_dim, int_dim_res))
        layers.append(Better_ResBlock(int_dim_res, int_dim_res))
        layers.append(Better_ResBlock(int_dim_res, int_dim_res))
        layers.append(Better_ResBlock(int_dim_res, int_dim_res))
        layers.append(nn.Linear(int_dim_res, int_dim_trf))
        layers.append(Better_Attention(int_dim_trf, n_channels))
        layers.append(Better_Transformer(int_dim_trf, n_channels))
        layers.append(Better_Attention(int_dim_trf, n_channels))
        layers.append(Better_Transformer(int_dim_trf, n_channels))
        layers.append(Better_Attention(int_dim_trf, n_channels))
        layers.append(Better_Transformer(int_dim_trf, n_channels))
        layers.append(nn.Linear(int_dim_trf,out_dim))
        layers.append(Affine())
        self.model = nn.Sequential(*layers)

        torch.set_default_tensor_type('torch.FloatTensor')

        self.model.load_state_dict(torch.load(PATH, map_location="cpu")())
        self.model.eval()

        with h5.File(PATH + '.h5', 'r') as f:
            self.X_mean = torch.Tensor(f['X_mean'][:])
            self.X_std  = torch.Tensor(f['X_std'][:])
            self.dv_fid = torch.Tensor(f['dv_fid'][:])
            self.dv_std = torch.Tensor(f['dv_std'][:])
            self.evecs  = torch.Tensor(f['evecs'][:])

        self.shear_calib_mask = np.load(os.environ.get("ROOTDIR") + '/external_modules/data/lsst_y1_cosmic_shear_emulator/shear_calib_mask.npy')[:,:780] 
        print('did theory finish initializing?')

    def predict(self, X):
        with torch.no_grad():
            y_pred = (self.model((torch.Tensor(X) - self.X_mean) / self.X_std) * self.dv_std)

        y_pred = y_pred @ torch.linalg.inv(self.evecs) + self.dv_fid
        return y_pred.cpu().detach().numpy()

    def add_shear_calib(self, m, datavector):
        for i in range(5):
            factor = (1 + m[i])**self.shear_calib_mask[i]
            factor = factor[0:780]
            datavector = factor * datavector
        return datavector

    def get_requirements(self):
        return {
          "logA": None,
          "H0": None,
          "ns": None,
          "omegabh2": None,
          "omegach2": None,
          "LSST_DZ_S1": None,
          "LSST_DZ_S2": None,
          "LSST_DZ_S3": None,
          "LSST_DZ_S4": None,
          "LSST_DZ_S5": None,
          "LSST_A1_1": None,
          "LSST_A1_2": None,
          "LSST_M1": None,
          "LSST_M2": None,
          "LSST_M3": None,
          "LSST_M4": None,
          "LSST_M5": None
        }

    def calculate(self, state, want_derived = True, **params):
        print('Am I calculating xi?')
        xi_params = [
            params['logA'],params['ns'],params['H0'],params['omegabh2'],params['omegach2'],
            params['LSST_DZ_S1'],params['LSST_DZ_S2'],params['LSST_DZ_S3'],params['LSST_DZ_S4'],params['LSST_DZ_S5'],
            params['LSST_A1_1'],params['LSST_A1_2'] 
        ]

        shear_calibration_params = [
            params['LSST_M1'],params['LSST_M2'],params['LSST_M3'],params['LSST_M4'],params['LSST_M5']
        ]

        print(xi_params)
        print(shear_calibration_params)

        xi_pm = self.predict(xi_params)[0]
        xi_pm = self.add_shear_calib(shear_calibration_params, xi_pm)

        state["lsst_y1_xi"] = xi_pm
        print('I calculated!')

        return True

    def get_can_support_params(self):
        return ['lsst_y1_xi']

    def get_lsst_y1_xi(self):
        return self.current_state['lsst_y1_xi']

    # def get_result(self,result_name):
    #     return self.current_state[result_name]




