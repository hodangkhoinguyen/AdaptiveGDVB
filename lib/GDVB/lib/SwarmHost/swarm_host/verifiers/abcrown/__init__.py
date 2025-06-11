import os

from .. import Verifier
from ..verifier_configs import VerifierConfigs


class ABCrown(Verifier):
    def __init__(self, verification_problem, configs):
        super().__init__(verification_problem)
        self.__name__ = "ABCrown"
        self.configs = configs

    
    def configure(self, config_path):
        '''
        vc = VerifierConfigs(self)
        if self.beta:
            vc.configs["solver"]["beta-crown"]["beta"] = True
        vc.save_configs(config_path)
        self.logger.debug(f"Verification config saved to: {config_path}")
        '''
        return
    

    def run(self, config_path, model_path, property_path, log_path, time):
        if self.configs['version'] == 22:
            cmd = f"$SwarmHost/scripts/run_abcrown22.sh"
        elif self.configs['version'] == 23:
            cmd = f"$SwarmHost/scripts/run_abcrown23.sh"
        elif self.configs['version'] == 24:
            cmd = f"$SwarmHost/scripts/run_abcrown24.sh"
        elif self.configs['version'] == 25:
            cmd = f"$SwarmHost/scripts/run_abcrown25.sh"
        else:
            assert False

        if not self.configs['beta']:
            cmd += ' --no_beta'
        else:
            pass

        # Always run with CUDA
        cmd += f" --device cuda --onnx_path {model_path} --vnnlib_path {property_path} --timeout {time} --config $SwarmHost/swarm_host/verifiers/abcrown/abcrown.yml"
        self.execute(cmd, log_path, time)

    def analyze(self):
        with open(self.verification_problem.paths["veri_log_path"], "r") as fp:
            lines = fp.readlines()
        lines = self.reformat_lines(lines)
        veri_ans, veri_time = super().pre_analyze(lines)
        iteration_count = 0

        if not (veri_ans and veri_time):
            veri_ans = None
            veri_time = None
            for l in lines:
                if "Result: " in l:
                    veri_ans = l.strip().split()[-1]
                elif "Time: " in l:
                    veri_time = float(l.strip().split()[-1])
                    
                elif 'CUDA out of memory' in l:
                    veri_ans = 'memout'
                    veri_time = -1

                # ERROR found by AdaGDVB
                elif "RuntimeError: cannot reshape tensor of 0 elements into shape [1, 0, -1] because the unspecified dimension size -1 can be any value and is ambiguous" in l:
                    veri_ans = 'error'
                    veri_time = -1
                elif "AttributeError: 'LiRPANet' object has no attribute 'split_indices'" in l:
                    veri_ans = 'error'
                    veri_time = -1
                elif "CUDA error: an illegal memory access was encountered" in l:
                    veri_ans = 'error'
                    veri_time = -1
                elif "TORCH_USE_CUDA_DSA" in l:
                    veri_ans = 'error'
                    veri_time = -1
                elif "RuntimeError: " in l:
                    veri_ans = 'memout'
                    veri_time = -1
                elif "BaB round" in l:
                    iteration_count += 1

                # if veri_ans and veri_time:
                #     break

        assert (
            veri_ans and veri_time
        ), f"Answer: {veri_ans}, time: {veri_time}, log: {self.verification_problem.paths['veri_log_path']}"
        
        return super().post_analyze(veri_ans, veri_time, iteration_count)
