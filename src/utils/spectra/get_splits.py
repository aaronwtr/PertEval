import os

import numpy as np

from src.utils.spectra.perturb import PerturbGraphData, SPECTRAPerturb


def spectra(sghv_pert_data, data_path, spectra_params, spectral_parameter):
    data_name = data_path.split('/')[-1]
    perturb_graph_data = PerturbGraphData(sghv_pert_data, data_name)

    sc_spectra = SPECTRAPerturb(perturb_graph_data, binary=False)
    sc_spectra.pre_calculate_spectra_properties(f"{data_path}/{data_name}")

    sparsification_step = spectra_params['sparsification_step']
    sparsification = ["{:.2f}".format(i) for i in np.arange(0, 1.01, float(sparsification_step))]
    spectra_params['number_repeats'] = int(spectra_params['number_repeats'])
    spectra_params['spectral_parameters'] = sparsification
    spectra_params['data_path'] = data_path + "/"

    if not os.path.exists(f"{data_path}/norman_SPECTRA_splits"):
        sc_spectra.generate_spectra_splits(**spectra_params)

    sp = spectral_parameter.split('_')[0]
    rpt = spectral_parameter.split('_')[1]
    train, test = sc_spectra.return_split_samples(sp, rpt,
                                                  f"{data_path}/{data_name}")
    pert_list = perturb_graph_data.samples

    return train, test, pert_list
