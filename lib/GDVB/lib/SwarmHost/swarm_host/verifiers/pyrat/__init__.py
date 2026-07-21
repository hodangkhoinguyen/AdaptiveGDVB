import os

from .. import Verifier


class PyRAT(Verifier):
    def __init__(self, verification_problem):
        super().__init__(verification_problem)
        self.__name__ = "PyRAT"

    def configure(self, config_path):
        ...

    def run(self, config_path, model_path, property_path, log_path, time):
        cmd = f"$SwarmHost/scripts/run_pyrat.sh"
        cmd += f" --model_path $ROOT/{model_path} --property_path $ROOT/{property_path}"
        cmd += f" --timeout {time} --domains poly"

        self.logger.info(f"Verifying: {cmd}")

        self.execute(cmd, log_path, time)

    def analyze(self):
        with open(self.verification_problem.paths["veri_log_path"], "r") as fp:
            lines = fp.readlines()

        veri_ans, veri_time = super().pre_analyze(lines)
        iteration_count = 0

        # PyRAT prints a summary line to stdout in the form:
        #   Result = <status>, Time = <seconds> s, Safe space = <pct> %, number of analysis = <n>
        # <status> is one of PyRATStatus's values (pyrat/config.py):
        #   True    -> property holds, no counterexample found -> unsat
        #   False   -> counterexample found, property violated -> sat
        #   Unknown -> analysis inconclusive
        #   Error   -> analysis raised an internal error
        #   Timeout -> PyRAT's own timeout was hit before resmonitor's
        status_map = {
            "True": "unsat",
            "False": "sat",
            "Unknown": "unknown",
            "Error": "error",
            "Timeout": "timeout",
        }

        if not (veri_ans and veri_time is not None):
            for l in lines:
                l = l.strip()
                if not l.startswith("Result = "):
                    continue
                parts = [p.strip() for p in l.split(",")]
                result_str = parts[0].split("=", 1)[1].strip()
                veri_ans = status_map.get(result_str)
                for p in parts[1:]:
                    if p.startswith("Time"):
                        veri_time = float(p.split("=", 1)[1].strip().rstrip("s").strip())
                    elif p.startswith("number of analysis"):
                        iteration_count = int(p.split("=", 1)[1].strip())
                if veri_ans and veri_time is not None:
                    break

        # NOTE: uses `is not None` rather than a plain truthiness check --
        # PyRAT rounds to 2 decimals and is often fast enough on small
        # networks to print "Time = 0.00 s", which is falsy but valid.
        assert (
            veri_ans and veri_time is not None
        ), f"Answer: {veri_ans}, time: {veri_time}, log: {self.verification_problem.paths['veri_log_path']}"

        return super().post_analyze(veri_ans, veri_time, iteration_count)
