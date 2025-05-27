import os

from .. import Verifier
from ..verifier_configs import VerifierConfigs


class MNBab(Verifier):
    def __init__(self, verification_problem):
        super().__init__(verification_problem)
        self.__name__ = "Mn-Bab"

    def configure(self, config_path):
        vc = VerifierConfigs(self)
        vc.save_configs(config_path)
        self.logger.debug(f"Verification config saved to: {config_path}")

    def run(self, config_path, model_path, property_path, log_path, time):

        cmd = f"$SwarmHost/scripts/run_mnbab.sh --config $ROOT/{config_path} --onnx_path $ROOT/{model_path} --vnnlib_path $ROOT/{property_path} --timeout {time}"
        
        self.execute(cmd, log_path, time)

    def analyze(self):
        with open(self.verification_problem.paths["veri_log_path"], "r") as fp:
            lines = fp.readlines()

        veri_ans, veri_time = super().pre_analyze(lines)

        if not (veri_ans and veri_time):
            veri_ans = None
            veri_time = None
            for l in lines[-100:]:
                if "Result: True" in l:
                    veri_ans = "unsat"
                elif "Result: False" in l:
                    veri_ans = "sat"
                elif "AssertionError: output_lb:" in l:
                    veri_ans = 'error'
                    veri_time = -1

                if "Time:" in l:
                    veri_time = float(l.split()[-1][:-1])

                error_pattern = [
                    "index_of_last_intermediate_bounds_kept",
                    "cannot reshape tensor of 0 elements into shape",
                    "Model was converted incorrectly",
                    "RuntimeError: mat1 and mat2 shapes cannot be multiplied",
                ]
                if any([True for x in error_pattern if x in l]):
                    veri_ans = "error"
                    veri_time = -1

                if veri_ans and veri_time:
                    break

        assert (
            veri_ans and veri_time
        ), f"Answer: {veri_ans}, time: {veri_time}, log: {self.verification_problem.paths['veri_log_path']}"
        
        return super().post_analyze(veri_ans, veri_time)
