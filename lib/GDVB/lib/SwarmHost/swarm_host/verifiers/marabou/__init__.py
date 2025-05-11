import os

from .. import Verifier
from ..verifier_configs import VerifierConfigs


class Marabou(Verifier):
    def __init__(self, verification_problem):
        print("Marabou(Verifier) run")
        super().__init__(verification_problem)
        self.__name__ = "Marabou"

    def configure(self, config_path):
        ...

    def run(self, config_path, model_path, property_path, log_path, time, memory):
        
        cmd = f"$SwarmHost/scripts/run_marabou.sh"
        cmd += f" {model_path} {property_path} --num-workers=64"
        self.logger.info(f'Verifying: {cmd}')
        
        self.execute(cmd, log_path, time, memory)

    def analyze(self):
        with open(self.verification_problem.paths["veri_log_path"], "r") as fp:
            lines = fp.readlines()
        veri_ans, veri_time = super().pre_analyze(lines)

        if not (veri_ans and veri_time):
            veri_ans = None
            veri_time = None
            for l in lines:
                if l.strip() == "sat":
                    veri_ans = "sat"
                elif l.strip() == "unsat":
                    veri_ans = "unsat"
                elif "Runtime:" in l:
                    veri_time = float(l.strip().split()[-1])
                elif "RuntimeError" in l:
                    veri_ans = "error"
                
                if veri_ans and veri_time:
                    break
        assert (
            veri_ans and veri_time
        ), f"Answer: {veri_ans}, time: {veri_time}, log: {self.verification_problem.paths['veri_log_path']}"
        return super().post_analyze(veri_ans, veri_time)
