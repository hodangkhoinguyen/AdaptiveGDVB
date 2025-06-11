import os

from .. import Verifier


class NNEnum(Verifier):
    def __init__(self, verification_problem):
        super().__init__(verification_problem)
        self.__name__ = "NNEnum"

    def configure(self, config_path):
        ...

    def run(self, config_path, model_path, property_path, log_path, time):
        
        cmd = f"$SwarmHost/scripts/run_nnenum.sh $ROOT/{model_path} $ROOT/{property_path} {time}"

        self.execute(cmd, log_path, time)

    def analyze(self):
        with open(self.verification_problem.paths["veri_log_path"], "r") as fp:
            lines = fp.readlines()

        veri_ans, veri_time = super().pre_analyze(lines)
        iteration_count = 0

        if not (veri_ans and veri_time):
            for l in lines:
                if "Result: network is SAFE" in l:
                    veri_ans = "unsat"
                elif "Result: network is UNSAFE with confirmed counterexample" in l:
                    veri_ans = "sat"

                if "Total Stars: " in l:
                    iteration_count = int(l.strip().split()[2])

                if "Runtime:" in l:
                    if "(" not in l:
                        veri_time = float(l.split()[-2])
                    else:
                        veri_time = float(l[str.index(l, "(") + 1 :].split()[0])

                if "reached during execution" in l:
                    veri_ans = "timeout"
                    veri_time = float(l[str.index(l, "(") + 1 : str.index(l, ")")])

                if "time limit has been exceeded" in l:
                    veri_ans = "timeout"
                    veri_time = -1

                error_pattern = [
                    "FloatingPointError: underflow encountered in multiply",
                    "underflow encountered in divide",
                    "FloatingPointError: overflow encountered in float_scalars",
                    "The search was prematurely terminated due to the solver"
                ]
                if any([True for x in error_pattern if x in l]):
                    veri_ans = "error"
                    veri_time = -1

                if veri_ans and veri_time:
                    break

        assert (
            veri_ans and veri_time
        ), f"Answer: {veri_ans}, time: {veri_time}, log: {self.verification_problem.paths['veri_log_path']}"
        
        return super().post_analyze(veri_ans, veri_time, iteration_count)
