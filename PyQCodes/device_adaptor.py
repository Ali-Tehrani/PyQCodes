from abc import ABC, abstractmethod


class DeviceAdaptorABC(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def unitary_two_design(self):
        pass

    @abstractmethod
    def estimate_average_fidelity(self):
        pass

    @abstractmethod
    def decomp_one_qubit_c_unitaries(self):
        # Project Q already implements this, so it is probably not useful at all.
        pass


class ProjectQDeviceAdaptor(DeviceAdaptorABC):
    def __init__(self):
        super(DeviceAdaptorABC).__init__()

    def unitary_two_design(self):
        pass

    def estimate_average_fidelity(self):
        pass

    def parameterize_circuits(self):
        pass

    def _compute_one_trial_average_fidelity(self):
        pass
