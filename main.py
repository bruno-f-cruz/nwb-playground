from datetime import datetime
from pathlib import Path
from uuid import uuid4

from aind_behavior_vr_foraging.data_contract import dataset
from pynwb import NWBHDF5IO, NWBFile

from nwb_formatter.models import Site
from nwb_formatter.processing import process_sites

dataset_path = Path(r"\\allen\aind\scratch\bruno.cruz\TestMouse_2026-01-24T011822Z")
ds = dataset(dataset_path)
processed_sites = process_sites(ds)


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
    nwbfile.add_trial(**site.model_dump())

print(nwbfile.trials.to_dataframe())

io = NWBHDF5IO("basics_tutorial.nwb", mode="w")
io.write(nwbfile)
io.close()
