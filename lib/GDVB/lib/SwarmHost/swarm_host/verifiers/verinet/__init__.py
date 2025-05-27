import os

from .. import Verifier
from ..verifier_configs import VerifierConfigs


class Verinet(Verifier):
    def __init__(self, verification_problem):
        super().__init__(verification_problem)
        self.__name__ = "Verinet"

        # TODO: add configuration?
        self.config_path = ""

    def configure(self, config_path):
        ...
    
    def run(self, config_path, model_path, property_path, log_path, time):
        
        input_shape = ' '.join(str(x) for x in self.verification_problem.property.shape)

        cmd = f"$SwarmHost/scripts/run_verinet.sh $ROOT/{model_path} $ROOT/{property_path} {time} --input_shape {input_shape}"
        
        print(cmd)
        self.execute(cmd, log_path, time)

    def analyze(self):
        with open(self.verification_problem.paths["veri_log_path"], "r") as fp:
            lines = fp.readlines()

        veri_ans, veri_time = super().pre_analyze(lines)

        if not (veri_ans and veri_time):
            for l in lines:
                if "Result: Status.Safe" in l:
                    veri_ans = "unsat"
                elif "Result: Status.Unsafe" in l:
                    veri_ans = "sat"

                if "Time: " in l:
                    veri_time = float(l.strip().split()[-1])

                '''
                error_pattern = [
                    "FloatingPointError: underflow encountered in multiply",
                    "underflow encountered in divide",
                ]
                if any([True for x in error_pattern if x in l]):
                    veri_ans = "error"
                    veri_time = -1
                '''
                if veri_ans and veri_time:
                    break

        assert (
            veri_ans and veri_time
        ), f"Answer: {veri_ans}, time: {veri_time}, log: {self.verification_problem.paths['veri_log_path']}"
        
        return super().post_analyze(veri_ans, veri_time)

