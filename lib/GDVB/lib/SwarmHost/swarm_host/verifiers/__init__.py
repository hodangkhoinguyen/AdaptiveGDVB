import os
import sys
import subprocess


class Verifier:
    def __init__(self, verification_problem):
        self.verification_problem = verification_problem
        self.logger = verification_problem.logger
        self.RES_MONITOR_PRETIME = 200

    def execute(self, cmd, log_path, time):
        res_monitor_path = os.path.join(os.environ["SwarmHost"], "lib", "resmonitor.py")
        cmd = (
            f"python3 {res_monitor_path} -T {time+self.RES_MONITOR_PRETIME} "
            + cmd
        )

        if log_path:
            veri_log_fp = open(log_path, "w")
        else:
            veri_log_fp = sys.stdout

        self.logger.info("Executing verification ...")
        self.logger.debug(cmd)
        self.logger.debug(f"Verification output path: {veri_log_fp}")

        sp = subprocess.Popen(cmd, shell=True, stdout=veri_log_fp, stderr=veri_log_fp)
        rc = sp.wait()
        assert rc == 0
        if log_path:
            veri_log_fp.close()

    def pre_analyze(self, lines):
        veri_ans = None
        veri_time = None

        for l in lines:
            if "Timeout (terminating process)" in l:
                veri_ans = "timeout"
                veri_time = float(l.strip().split()[-1])
            elif "Out of Memory" in l:
                veri_ans = "memout"
                veri_time = -1
            elif "Model does not exist" in l:
                veri_ans = 'hardware_limit'
                veri_time = -1
            elif "exceeds maximum protobuf size of 2GB" in l:
                veri_ans = 'hardware_limit'
                veri_time = -1

        return veri_ans, veri_time

    def post_analyze(self, answer, time):
        if answer != 'timeout' and time > self.verification_problem.verifier_config["time"]:
            answer = 'timeout'
            time = self.verification_problem.verifier_config["time"]
        return answer, time

    """
    Helper function:
        the logger print out in between line of logging file which affect the analysis process
        rewrite lines to put INFO into a new line
    """
    def reformat_lines(self, lines):
        new_lines = []
        i = 0
        size = len(lines)
        while i < size:
            line = lines[i]
            if "INFO" in line and "(resmonitor)" in line and not line.startswith("INFO"):
                curr = line
                regular, info = curr.split("INFO     ")
                info = "INFO     " + info
                i += 1
                new_lines.append(info)
                while lines[i].startswith("INFO"):
                    new_lines.append(lines[i])
                    i += 1
                regular += lines[i]
                new_lines.append(regular)
                i += 1
                continue
            new_lines.append(line)
            i += 1

        return new_lines
