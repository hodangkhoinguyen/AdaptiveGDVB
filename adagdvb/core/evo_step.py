import os
import sys
import time
import numpy as np
import pickle

from enum import Enum, auto

from pathlib import Path
from tqdm import tqdm


from .factor import Factor

from gdvb.core.verification_benchmark import VerificationBenchmark
from gdvb.plot.pie_scatter import PieScatter2D

TIME_BREAK = 1


class EvoStep:
    class Direction(Enum):
        Both = auto()
        Up = auto()
        Down = auto()
        Maintain = auto()

    def __init__(
        self,
        benchmark,
        evo_params,
        direction,
        iteration,
        logger,
        critical_region_analysis,
        success_answers=("sat", "unsat"),
    ):
        self.logger = logger
        self.benchmark = benchmark
        self.evo_params = evo_params
        self.critical_region_analysis = critical_region_analysis
        self.iteration = iteration
        self.direction = direction
        self.nb_solved = None
        self.answers = None
        # which conclusive verifier answers count as "solved" for VPB search
        # purposes. Default (sat or unsat) matches the original behavior; set
        # to ("unsat",) to focus the boundary on proving difficulty only,
        # since sat instances (falsified by finding a counterexample) tend to
        # resolve quickly regardless of network size and so dilute the
        # solve-rate signal used to locate the boundary.
        self.success_answers = tuple(success_answers)
        # TODO: only support one verifier at a time
        self.verifier = list(
            benchmark.settings.verification_configs["verifiers"].values()
        )[0][0]
        self.factors = self._gen_factors()

    def _gen_factors(self):
        factors = []
        for p in self.evo_params:
            start = self.benchmark.ca_configs["parameters"]["range"][p][0]
            end = self.benchmark.ca_configs["parameters"]["range"][p][1]
            level = self.benchmark.ca_configs["parameters"]["level"][p]
            fc_conv_ids = {"fc": self.benchmark.fc_ids, "conv": self.benchmark.conv_ids}
            factors += [Factor(p, start, end, level, fc_conv_ids)]
        return factors

    def forward(self):
        # launch training jobs
        self.benchmark.train()

        # wait for training
        nb_train_tasks = len(self.benchmark.verification_problems)
        progress_bar = tqdm(
            total=nb_train_tasks,
            desc="Waiting on training ... ",
            ascii=False,
            file=sys.stdout,
        )
        nb_trained_pre = self.benchmark.trained(True)

        progress_bar.update(nb_trained_pre)
        while not self.benchmark.trained():
            time.sleep(TIME_BREAK)
            nb_trained_now = self.benchmark.trained(True)
            progress_bar.update(nb_trained_now - nb_trained_pre)
            progress_bar.refresh()
            nb_trained_pre = nb_trained_now
        progress_bar.close()

        # analyze training results
        self.benchmark.analyze_training()

        # execute critical region analysis
        if self.critical_region_analysis:
            self.benchmark.critical_region_analysis()

        # launch verification jobs
        self.benchmark.verify()

        # wait for verification
        nb_verification_tasks = len(self.benchmark.verification_problems)
        progress_bar = tqdm(
            total=nb_verification_tasks,
            desc="Waiting on verification ... ",
            ascii=False,
            file=sys.stdout,
        )

        nb_verified_pre = self.benchmark.verified(True)
        progress_bar.update(nb_verified_pre)
        while not self.benchmark.verified():
            time.sleep(TIME_BREAK)
            nb_verified_now = self.benchmark.verified(True)
            progress_bar.update(nb_verified_now - nb_verified_pre)
            progress_bar.refresh()
            nb_verified_pre = nb_verified_now
        progress_bar.close()

        # analyze verification results
        self.benchmark.analyze_verification()

    # process verification results for things
    def evaluate(self):
        benchmark = self.benchmark
        ca_configs = benchmark.ca_configs
        indexes = {}
        for p in self.evo_params:
            ids = []
            for vpc in benchmark.ca:
                ids += [vpc[x] for x in vpc if x == p]
            indexes[p] = sorted(set(ids))

        nb_property = ca_configs["parameters"]["level"]["prop"]
        solved_per_verifiers = {}
        answers_per_verifiers = {}
        times_per_verifiers = {}
        for problem in benchmark.verification_problems:
            for verifier in problem.verification_results:
                if verifier not in solved_per_verifiers:
                    shape = ()
                    for p in self.evo_params:
                        shape += (ca_configs["parameters"]["level"][p],)
                    solved_per_verifiers[verifier] = np.zeros(shape, dtype=np.int32)
                    answers_per_verifiers[verifier] = np.empty(
                        shape + (nb_property,), dtype=np.int32
                    )
                    times_per_verifiers[verifier] = np.zeros(shape, dtype=np.float32)

                idx = tuple(indexes[x].index(problem.vpc[x]) for x in self.evo_params)
                if problem.verification_results[verifier][0] in self.success_answers:
                    solved_per_verifiers[verifier][idx] += 1
                times_per_verifiers[verifier][idx] += problem.verification_results[
                    verifier
                ][1] / nb_property
                prop_id = problem.vpc["prop"]
                answer_code = benchmark.settings.answer_code[
                    problem.verification_results[verifier][0]
                ]
                answers_per_verifiers[verifier][idx + (prop_id,)] = answer_code

        self.nb_solved = solved_per_verifiers
        self.answers = answers_per_verifiers
        self.times = times_per_verifiers

    def _get_cache_prefix(self):
        self.logger.info("Loading verification cache ...")
        cache_dir = os.path.join(self.benchmark.settings.root, f"cache_{self.verifier}")
        Path(cache_dir).mkdir(exist_ok=True, parents=True)
        cache_path = os.path.join(cache_dir, f"{self.iteration}_{self.direction}")
        if self.critical_region_analysis:
            cache_path += "_CRA"

        return cache_path

    def save_cache(self):
        self.logger.info("Saving results cache ...")
        cache_prefix = self._get_cache_prefix()
        cache_solved_path = f"{cache_prefix}_solved.pkl"
        with open(cache_solved_path, "wb") as f:
            pickle.dump(self.nb_solved, f)
        cache_answers_path = f"{cache_prefix}_answers.pkl"
        with open(cache_answers_path, "wb") as f:
            pickle.dump(self.answers, f)
        cache_times_path = f"{cache_prefix}_times.pkl"
        with open(cache_times_path, "wb") as f:
            pickle.dump(self.times, f)

    def load_cache(self):
        self.logger.info("Loading results cache ...")
        cache_prefix = self._get_cache_prefix()

        cache_solved_path = f"{cache_prefix}_solved.pkl"
        cache_answers_path = f"{cache_prefix}_answers.pkl"
        cache_times_path = f"{cache_prefix}_times.pkl"

        if (
            os.path.exists(cache_solved_path)
            and os.path.exists(cache_answers_path)
            and os.path.exists(cache_times_path)
        ):
            cache_hit = True
            with open(cache_solved_path, "rb") as f:
                self.nb_solved = pickle.load(f)
            with open(cache_answers_path, "rb") as f:
                self.answers = pickle.load(f)
            with open(cache_times_path, "rb") as f:
                self.times = pickle.load(f)
        else:
            cache_hit = False

        return cache_hit

    def plot(self):
        if len(self.evo_params) == 2:
            # TODO: only supports one([0]) verifier per time
            data = list(self.answers.values())[0]

            labels = self.evo_params
            ticks = [
                np.array(x.explicit_levels, dtype=np.float32).tolist()
                for x in self.factors
            ]

            # print('XXXXXXXXXXXXXXXXX', set(sorted([np.array(x.explicit_levels).tolist() for x in self.factors][0])))
            # print('XXXXXXXXXXXXXXXXX', set(sorted([np.array(x.explicit_levels).tolist() for x in self.factors][1])))

            x_ticks = [f"{x:.4f}" for x in ticks[0]]
            y_ticks = [f"{x:.4f}" for x in ticks[1]]
            pie_scatter = PieScatter2D(data)
            pie_scatter.draw(x_ticks, y_ticks, labels[0], labels[1])
            # pdf_dir = f'./img/{list(self.answers.keys())[0]}'
            pdf_dir = f"{self.benchmark.settings.root}/figures/"
            Path(pdf_dir).mkdir(parents=True, exist_ok=True)
            pie_scatter.save(f"{pdf_dir}/{self.iteration}_{self.direction}.png")

            #data = list(self.answers.values())[1]

        else:
            raise NotImplementedError

    def __str__(self) -> str:
        res = f"Iter:\t{self.iteration}"
        res += f"Dir:\t{self.direction}"
        for p in self.evo_params:
            res += f"{p}:\t{[f'{x:.3f}' for x in sorted(set([x[p] for x in self.benchmark.ca]))]}\n"
        return res
