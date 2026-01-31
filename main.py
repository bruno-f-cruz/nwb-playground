from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
from aind_behavior_vr_foraging.data_contract import dataset
from pynwb import NWBFile

from nwb_formatter.models import Site
from nwb_formatter.processing import DatasetProcessor

dataset_path = Path(r"\\allen\aind\stage\vr-foraging\data\828424\828424_2026-01-31T001737Z")
ds = dataset(dataset_path)
processed_sites = DatasetProcessor(ds).process()


# Create a new NWBFile. Most of these are already available via the aind-data-schema
nwbfile = NWBFile(
    session_description="Foo",
    identifier=str(uuid4()),
    session_start_time=datetime.now().astimezone(),
    experimenter=[
        "Baggins, Bilbo",
    ],
    lab="Bag End Laboratory",
    institution="University of Middle Earth at the Shire",
    experiment_description="I went on an adventure to reclaim vast treasures.",
    session_id="LONELYMTN001",
)

for field_name, field in Site.model_fields.items():
    if field_name in ["start_time", "stop_time"]:
        continue
    nwbfile.add_trial_column(name=field_name, description=field.description)

for site in processed_sites:
    trial_data = site.model_dump()
    # Replace None with np.nan
    trial_data = {k: (np.nan if v is None else v) for k, v in trial_data.items()}
    nwbfile.add_trial(**trial_data)

a = nwbfile.trials.to_dataframe()
rewarded_sites = a[a["site_label"] == "RewardSite"]
for patch_id in rewarded_sites["patch_label"].unique():
    patch_data = rewarded_sites[rewarded_sites["patch_label"] == patch_id]
    p_choice = patch_data["has_choice"].mean()
    p_reward = patch_data["has_reward"].sum() / len(patch_data)
    print(f"Patch {patch_id}: P(choice)={p_choice:.2f}, P(reward|choice)={p_reward:.2f}")

# io = NWBHDF5IO("basics_tutorial.nwb", mode="w")
# io.write(nwbfile)
# io.close()
