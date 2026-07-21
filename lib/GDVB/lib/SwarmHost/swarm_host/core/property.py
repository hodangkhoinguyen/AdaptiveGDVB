import os
import onnxruntime as ort
import numpy as np
import onnx


import numpy
from torchvision import datasets, transforms
from pathlib import Path

class Property:
    def __init__(self, logger):
        self.logger = logger

    def set(self, path):
        self.property_path = path


class LocalRobustnessProperty(Property):
    def __init__(self, logger, property_configs):
        super().__init__(logger)
        self.property_configs = property_configs

    def generate(self, prop_dir, format, model_path=None):
        if format == "vnnlib":
            self.gen_vnnlib(prop_dir, model_path)
        else:
            raise NotImplementedError()

    @staticmethod
    def _to_model_input(img_npy):
        if len(img_npy.shape) == 2:
            return img_npy.reshape(1, 1, *img_npy.shape)
        elif len(img_npy.shape) == 3:
            return img_npy.reshape(1, *img_npy.shape)
        elif len(img_npy.shape) == 4:
            return img_npy
        else:
            raise NotImplementedError()

    def _select_margin_aware_id(self, test_dataset, model_path):
        # Ranks a candidate pool of images by this specific trained network's
        # own clean-input classification margin (top-1 logit minus runner-
        # up), then picks the `id`-th (the CA-assigned prop level) image from
        # a target quantile band of that ranking, instead of `id` indexing
        # the raw dataset directly. This calibrates property difficulty per
        # trained network -- every (neu, fc) grid point in AdaGDVB trains its
        # own distinct distilled network -- rather than leaving it to
        # whichever image an arbitrary dataset index happens to land on,
        # which is what makes solve rate noisy across property samples at a
        # fixed architecture coordinate today.
        pool_size = min(
            self.property_configs.get("margin_pool_size", 100), len(test_dataset)
        )
        band = self.property_configs.get("margin_band", (0.0, 0.3))

        session = ort.InferenceSession(model_path)
        input_names = [x.name for x in session.get_inputs()]
        output_names = [x.name for x in session.get_outputs()]
        assert len(input_names) == 1 and len(output_names) == 1

        margins = []
        for idx in range(pool_size):
            img, _ = test_dataset[idx]
            img_npy = self._to_model_input(numpy.asarray(img))
            logits = np.asarray(
                session.run(output_names, {input_names[0]: img_npy})
            ).reshape(-1)
            top2 = np.sort(logits)[-2:]
            margins += [float(top2[-1] - top2[-2])]

        # ascending: order[0] is the smallest margin (hardest/most borderline)
        order = np.argsort(margins)
        lo_q, hi_q = band
        lo_i = int(round(lo_q * (len(order) - 1)))
        hi_i = int(round(hi_q * (len(order) - 1)))
        banded = order[lo_i : hi_i + 1]
        if len(banded) == 0:
            banded = order

        # `id` still ranges over 0..nb_property-1 (the CA-assigned prop
        # level); it now selects deterministically within the band instead
        # of indexing the dataset directly, so distinct prop levels still
        # map to distinct (now difficulty-calibrated) images -- modulo
        # repeats if the band ends up narrower than nb_property.
        prop_level = self.property_configs["id"]
        return int(banded[prop_level % len(banded)])

    def gen_vnnlib(self, prop_dir, model_path=None):
        artifact = self.property_configs["artifact"]
        eps = self.property_configs["eps"]
        Path(prop_dir).mkdir(exist_ok=True,parents=True)

        t = [transforms.ToTensor()]

        mean = self.property_configs['mean']
        std = self.property_configs['std']
        if mean and std:
            t += [transforms.Normalize(mean, std)]
        elif not mean and not std:
            t += [transforms.Normalize((0,), (1,))]
        else:
            assert False,"mean and std must be configured the same time"
        transform = transforms.Compose(t)
        test_dataset = eval(f"datasets.{artifact}")(
            "data", download=True, train=False, transform=transform
        )

        if self.property_configs.get("margin_aware"):
            assert model_path, "margin_aware property selection requires model_path"
            img_id = self._select_margin_aware_id(test_dataset, model_path)
        else:
            img_id = self.property_configs["id"]

        self.property_path = os.path.join(prop_dir, f"{artifact}_{img_id}_{eps}.vnnlib")
        img, label = test_dataset[img_id]
        img_npy = numpy.asarray(img)
        self.shape = img_npy.shape
        img_npy_flatten = img_npy.flatten()
        
        if self.property_configs["mrb"]:
            assert  model_path
            
            # model = onnx.load(model_path, load_external_data=False).SerializeToString()
            # model = onnx.load(model_path, load_external_data=True).SerializeToString()
            session = ort.InferenceSession(model_path)
            #names = [i.name for i in sess.get_inputs()]
            #label= sess.run(None, dict(zip(names, img_npy)))
            session.get_modelmeta()
            
            input_names = [x.name for x in session.get_inputs()]
            output_names = [x.name for x in session.get_outputs()]
            
            assert len(input_names) == 1 and len(output_names) == 1
            img_npy = self._to_model_input(img_npy)

            print(img_npy.shape)
            #results = session.run([output_names[0]], {input_names[0]: img_npy})
            results = session.run(output_names, {input_names[0]:img_npy})

            pred = np.argmax(results)
            
            label=pred

        # generate VNN-lib Property
        # 1) define input
        vnn_lib_lines = [f"; {artifact} property with label: {label}.", ""]

        for x in range(len(img_npy_flatten)):
            vnn_lib_lines += [f"(declare-const X_{x} Real)"]

        # 2) define output
        vnn_lib_lines += [""]
        if artifact in ["MNIST", "CIFAR10"]:
            nb_output = 10
        elif artifact == "DAVE2":
            nb_output = 1
        else:
            assert False
        for x in range(nb_output):
            vnn_lib_lines += [f"(declare-const Y_{x} Real)"]

        # 3) define input constraints:
        vnn_lib_lines += ["", "; Input constraints:"]
        for i, x in enumerate(img_npy_flatten):
            lb = x - eps
            ub = x + eps
            if "clip" in self.property_configs and self.property_configs["clip"]:
                if lb < 0:
                    lb = 0.0
                if ub > 1:
                    ub = 1.0
            vnn_lib_lines += [
                f"(assert (<= X_{i} {ub}))",
                f"(assert (>= X_{i} {lb}))",
            ]

        # 4) define output constraints:
        vnn_lib_lines += ["", f"; Output constraints:"]
        if artifact in ["MNIST", "CIFAR10"]:
            vnn_lib_lines += ["(assert (or"]
            for x in range(nb_output):
                if not x == label:
                    vnn_lib_lines += [f"    (and (>= Y_{x} Y_{label}))"]
            vnn_lib_lines += ["))"]

        elif artifact == "DAVE2":
            assert False
            lines = open(
                os.path.join(os.path.dirname(img_path), "properties.csv"), "r"
            ).readlines()

            for x in lines[1:]:
                tokens = x.split(",")
                if img_id == int(tokens[0]):
                    min_val = tokens[-2]
                    max_val = tokens[-1]
                    break
            assert min_val and max_val
            vnn_lib_lines += ["(assert (or"]
            vnn_lib_lines += [f"(and (>= Y_0 {min_val}))"]
            vnn_lib_lines += [f"(and (<= Y_0 {max_val})"]
            vnn_lib_lines += ["))"]
        else:
            assert False

        with open(self.property_path, "w") as fp:
            fp.writelines(x + "\n" for x in vnn_lib_lines)
        self.logger.debug(f"Property generated: {self.property_path}")