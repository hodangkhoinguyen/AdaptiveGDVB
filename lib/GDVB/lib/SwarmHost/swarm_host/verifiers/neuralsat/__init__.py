import os

from .. import Verifier
from ..verifier_configs import VerifierConfigs


class NeuralSat(Verifier):
    def __init__(self, verification_problem, version):
        super().__init__(verification_problem)
        self.version = version
        self.__name__ = "NeuralSat"

    def configure(self, config_path):
        ...

    def run(self, config_path, model_path, property_path, log_path, time):
        
        cmd = f"$SwarmHost/scripts/run_neuralsat.sh"
        if self.version == 1:
            cmd += f" --batch 1 --disable_restart --disable_stabilize"
        elif self.version == 2:
            cmd += f" --batch 1 --disable_restart"
        elif self.version == 3:
            pass
        else:
            assert False
        cmd += f" --net {model_path} --spec {property_path}"
        self.logger.info(f'Verifying: {cmd}')
        
        self.execute(cmd, log_path, time)

    def analyze(self):
        with open(self.verification_problem.paths["veri_log_path"], "r") as fp:
            lines = fp.readlines()
        veri_ans, veri_time = super().pre_analyze(lines)
        iteration_count = 0

        if not (veri_ans and veri_time):
            veri_ans = None
            veri_time = None
            for l in lines:
                if "AssertionError" in l:
                    veri_ans = 'error'
                    veri_time = -1
                elif "CUDA error: out of memory" in l:
                    veri_ans = 'memout'
                    veri_time = -1
                elif "CUDA out of memory" in l:
                    veri_ans = 'memout'
                    veri_time = -1
                # elif "RuntimeError" in l:
                #     veri_ans = 'error'
                #     veri_time = -1
                elif "Gurobi error: Model too large for size-limited license" in l:
                    veri_ans = 'memout'
                    veri_time = -1
                elif "[!] Result:" in l:
                    veri_ans = l.strip().split()[-1]
                elif "[!] Runtime:" in l:
                    veri_time = float(l.strip().split()[-1])
                elif "[!] Iterations:" in l:
                    iteration_count = int(l.strip().split()[-1])
 
                # if veri_ans and veri_time:
                #     break
        assert (
            veri_ans and veri_time
        ), f"Answer: {veri_ans}, time: {veri_time}, log: {self.verification_problem.paths['veri_log_path']}"
        return super().post_analyze(veri_ans, veri_time, iteration_count)

'''
class NeuralSatP(Verifier):
    def __init__(self, verification_problem):
        super().__init__(verification_problem)
        self.__name__ = "NeuralSatP"

    def configure(self, config_path):
        ...

    def run(self, config_path, model_path, property_path, log_path, time, memory):
        
        cmd = f"$SwarmHost/scripts/run_neuralsat.sh --batch 1 --disable_restart --net {model_path} --spec {property_path}"
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
                if "AssertionError" in l:
                    veri_ans = 'error'
                    veri_time = -1
                elif "[!] Result:" in l:
                    veri_ans = l.strip().split()[-1]
                elif "[!] Runtime:" in l:
                    veri_time = float(l.strip().split()[-1])
                
                if veri_ans and veri_time:
                    break
        assert (
            veri_ans and veri_time
        ), f"Answer: {veri_ans}, time: {veri_time}, log: {self.verification_problem.paths['veri_log_path']}"
        return super().post_analyze(veri_ans, veri_time)
'''