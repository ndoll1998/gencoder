import os
import _io
import json
import datasets
import multiprocessing as mp
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
from torch.utils.data._utils.worker import get_worker_info
from hyped.data.io.writers.base import BaseDatasetWriter
from hyped.common.arrow import convert_features_to_arrow_schema


class ArrowDatasetWriter(BaseDatasetWriter):

    def __init__(self, *args, **kwargs):
        super(ArrowDatasetWriter, self).__init__(*args, **kwargs)

        self._data_files = mp.Manager().list()

    def consume(
        self, data: datasets.Dataset | datasets.IterableDataset
    ) -> None:
         
        # consume dataset
        super(ArrowDatasetWriter, self).consume(data)

        dataset_info = asdict(data.info)
        state = {
            key: getattr(data, key, None)
            for key in [
                "_fingerprint",
                "_format_columns",
                "_format_kwargs",
                "_format_type",
                "_output_all_columns",
            ]
        }
        state["_format_kwargs"] = {}
        state["_split"] = str(data.split) if data.split is not None else data.split
        state["_data_files"] = [{"filename": os.path.basename(fpath)} for fpath in self._data_files]

        for k in state["_format_kwargs"].keys():
            try:
                json.dumps(state["_format_kwargs"][k])
            except TypeError as e:
                raise TypeError(
                    str(e) + f"\nThe format kwargs must be JSON serializable, but key '{k}' isn't."
                ) from None

        with open(
            os.path.join(self.save_dir, datasets.config.DATASET_STATE_JSON_FILENAME), "w", encoding="utf-8"
        ) as state_file:
            json.dump(state, state_file, indent=2, sort_keys=True)
        with open(
            os.path.join(self.save_dir, datasets.config.DATASET_INFO_FILENAME), "w", encoding="utf-8"
        ) as dataset_info_file:
            # Sort only the first level of keys, or we might shuffle fields of nested features if we use sort_keys=True
            sorted_keys_dataset_info = {key: dataset_info[key] for key in sorted(dataset_info)}
            json.dump(sorted_keys_dataset_info, dataset_info_file, indent=2)

    def worker_shard_file_obj(
        self, path: str, worker_id: int
    ) -> _io.TextIOWrapper:
        return open(os.path.join(path, "worker_shard_%i.arrow" % worker_id), "wb")

    def initialize_worker(self):
        super(ArrowDatasetWriter, self).initialize_worker()

        info = get_worker_info()
        info.args.schema = convert_features_to_arrow_schema(info.dataset.features)
        info.args.writer = pa.ipc.new_stream(
            sink=info.args.save_file,
            schema=info.args.schema
        )

    def finalize_worker(self):
        # close writer
        info = get_worker_info()
        info.args.writer.close()
        # close file
        super(ArrowDatasetWriter, self).finalize_worker()
        
        if os.path.isfile(info.args.save_file_path):
            self._data_files.append(info.args.save_file_path)

    def consume_example(
        self,
        shard_id: int,
        example_id: int,
        example: dict[str, Any],
    ) -> None:
        info = get_worker_info()
        example = pa.RecordBatch.from_pylist(
            [example], schema=info.args.schema
        )
        info.args.writer.write(example)
