import logging
import typing as t

import aind_behavior_vr_foraging.task_logic as vrf_task
import contraqctor
import numpy as np
import pandas as pd

from .models import Site, Trial

logger = logging.getLogger(__name__)


def parse_trials(dataset: contraqctor.contract.Dataset) -> pd.DataFrame:
    reward_label = vrf_task.VirtualSiteLabels.REWARDSITE
    rewarded_sites = dataset.at("Behavior").at("SoftwareEvents").at("ActiveSite").load().data.copy()
    rewarded_sites = rewarded_sites[rewarded_sites["data"].apply(lambda d: d["label"] == reward_label)]

    # Merge nearest patch (backward in time)
    merged = pd.merge_asof(
        rewarded_sites,
        dataset.at("Behavior").at("SoftwareEvents").at("ActivePatch").load().data[["data"]],
        left_index=True,
        right_index=True,
        direction="backward",
        suffixes=("", "_patch"),
    )

    merged.rename(columns={"data_patch": "patches"}, inplace=True)
    merged["patch_index"] = merged.patches.apply(lambda d: d["state_index"])

    speaker_choice = dataset.at("Behavior").at("HarpBehavior").load().at("PwmStart").load().data.copy()
    speaker_choice = speaker_choice[(speaker_choice["MessageType"] == "WRITE") & (speaker_choice["PwmDO2"])]

    water_delivery = dataset.at("Behavior").at("HarpBehavior").load().at("OutputSet").load().data.copy()
    water_delivery = water_delivery[(water_delivery["MessageType"] == "WRITE") & (water_delivery["SupplyPort0"])][
        "SupplyPort0"
    ]

    odor_onset = dataset.at("Behavior").at("HarpOlfactometer").load().at("EndValveState").load().data
    odor_onset = odor_onset[odor_onset["MessageType"] == "WRITE"]["EndValve0"]
    odor_onset = odor_onset[(odor_onset) & (~odor_onset.shift(1, fill_value=False))]

    patches_state = dataset.at("Behavior").at("SoftwareEvents").at("PatchState").load().data.copy()
    expanded = pd.json_normalize(patches_state["data"])
    expanded.index = patches_state.index
    patches_state = patches_state.join(expanded)

    patches_state_at_reward = dataset.at("Behavior").at("SoftwareEvents").at("PatchStateAtReward").load().data.copy()
    expanded = pd.json_normalize(patches_state_at_reward["data"])
    expanded.index = patches_state_at_reward.index
    patches_state_at_reward = patches_state_at_reward.join(expanded)

    trials: list[Trial] = []

    i = 0
    for i in range(len(merged) - 1):
        this_timestamp = merged.index[i]
        next_timestamp = merged.index[i + 1]
        logger.debug(f"Processing trial {i} at {this_timestamp} - {next_timestamp}")
        ## Find closest odor_onset after this_timestamp but before next_timestamp
        odor_onsets_in_interval = odor_onset[(odor_onset.index >= this_timestamp) & (odor_onset.index < next_timestamp)]
        if len(odor_onsets_in_interval) == 0:
            logger.warning(f"No odor onset in site {i} interval...Using software event instead")
            odor_onsets_in_interval = merged.loc[[this_timestamp]]

        ## Find closest speaker_choice after this_timestamp but before next_timestamp
        speaker_choices_in_interval = speaker_choice[
            (speaker_choice.index >= this_timestamp) & (speaker_choice.index < next_timestamp)
        ]
        assert len(speaker_choices_in_interval) <= 1, "Multiple speaker choices in interval"

        stops = dataset.at("Behavior").at("OperationControl").at("IsStopped").data
        ## Find the closest is stop
        if len(speaker_choices_in_interval) > 0:
            mask = (stops.index <= speaker_choices_in_interval.index[0]) & (stops.iloc[:, 0])
            stops_before_speaker = stops.loc[mask]
            if len(stops_before_speaker) > 0:
                stop_time = stops_before_speaker.index[-1]  # Get the closest one before
            else:
                raise ValueError("No stop found before speaker choice")
        else:
            stop_time = None

        ## Find the longest stop inside the interval
        stop_data_inside_interval = stops[(stops.index >= this_timestamp) & (stops.index < next_timestamp)]
        if not stop_data_inside_interval.empty:
            # If the first value is True, we compute the duration from the odor onset
            if stop_data_inside_interval.iloc[0, 0]:
                prepend = pd.DataFrame(
                    [[False]], index=[odor_onsets_in_interval.index[0]], columns=stop_data_inside_interval.columns
                )
                stop_data_inside_interval = pd.concat([prepend, stop_data_inside_interval])

            index_diff = np.diff(stop_data_inside_interval.index.values)
            mask = stop_data_inside_interval.values.flatten()
            longest_stop_duration = index_diff[mask[:-1]].max() if len(index_diff[mask[:-1]]) > 0 else None
        else:
            longest_stop_duration = None

        ## Find closest water_delivery after this_timestamp but before next_timestamp
        water_deliveries_in_interval = water_delivery[
            (water_delivery.index >= this_timestamp) & (water_delivery.index < next_timestamp)
        ]
        if len(water_deliveries_in_interval) > 1:
            logger.warning(f"Multiple water deliveries in interval {this_timestamp} - {next_timestamp}")
            water_deliveries_in_interval = water_deliveries_in_interval.iloc[:1]

        # Get the FIRST patch state AFTER the this_timestamp
        site_state_at_reward = patches_state_at_reward[
            (patches_state_at_reward.index > this_timestamp)
            & (patches_state_at_reward["PatchId"] == merged.iloc[i]["patch_index"])
        ]
        if len(site_state_at_reward) > 0:
            site_state_at_reward = site_state_at_reward.iloc[0]
            # TODO this is because of block switches...
            trial = Trial(
                odor_onset_time=odor_onsets_in_interval.index[0],
                choice_time=speaker_choices_in_interval.index[0] if len(speaker_choices_in_interval) == 1 else None,
                reward_time=water_deliveries_in_interval.index[0] if len(water_deliveries_in_interval) == 1 else None,
                reaction_duration=(speaker_choices_in_interval.index[0] - odor_onsets_in_interval.index[0])
                if len(speaker_choices_in_interval) == 1
                else None,
                patch_index=merged.iloc[i]["patch_index"],
                is_rewarded=len(water_deliveries_in_interval) == 1,
                p_reward=site_state_at_reward["Probability"],
                is_choice=len(speaker_choices_in_interval) == 1,
                begin_stop_time=stop_time,
                longest_stop_duration=longest_stop_duration,
                stop_time=next_timestamp,
                start_time=this_timestamp,
            )
            trials.append(trial)
        else:
            trials.append(
                Trial(
                    odor_onset_time=odor_onsets_in_interval.index[0],
                    choice_time=None,
                    reward_time=None,
                    reaction_duration=None,
                    patch_index=merged.iloc[i]["patch_index"],
                    is_rewarded=None,
                    p_reward=np.nan,
                    is_choice=False,
                    longest_stop_duration=None,
                    stop_time=next_timestamp,
                    start_time=this_timestamp,
                    begin_stop_time=None,
                )
            )

    trials_df = pd.DataFrame([trial.__dict__ for trial in trials])
    return trials_df


def get_closest_from_timestamp(
    timestamps: np.ndarray,
    df: pd.DataFrame,
    *,
    search_mode: t.Literal["closest", "next", "previous"] = "closest",
) -> np.ndarray:
    """
    For each timestamp in `timestamps`, find the index in df.index that is:
      - 'closest': closest in value
      - 'next': the first index >= timestamp
      - 'previous': the last index <= timestamp

    Returns an array of indices from df.index.
    """
    df_index = df.index.values

    # Use numpy searchsorted for efficient lookup
    timestamps = np.asarray(timestamps)
    if search_mode == "closest":
        idx_left = np.searchsorted(df_index, timestamps, side="left")
        idx_right = np.clip(idx_left - 1, 0, len(df_index) - 1)
        idx_left = np.clip(idx_left, 0, len(df_index) - 1)
        left_diff = np.abs(df_index[idx_left] - timestamps)
        right_diff = np.abs(df_index[idx_right] - timestamps)
        use_left = left_diff <= right_diff
        idxs = np.where(use_left, idx_left, idx_right)
    elif search_mode == "next":
        idxs = np.searchsorted(df_index, timestamps, side="left")
        idxs = np.clip(idxs, 0, len(df_index) - 1)
    elif search_mode == "previous":
        idxs = np.searchsorted(df_index, timestamps, side="right") - 1
        idxs = np.clip(idxs, 0, len(df_index) - 1)
    else:
        raise ValueError(f"Unknown search_mode: {search_mode}")
    return df.index[idxs]


def parse_speaker_choice_feedback(dataset: contraqctor.contract.Dataset) -> pd.DataFrame:
    speaker_choice = dataset.at("Behavior").at("HarpBehavior").load().at("PwmStart").load().data.copy()
    speaker_choice = speaker_choice[(speaker_choice["MessageType"] == "WRITE") & (speaker_choice["PwmDO2"])]
    return speaker_choice


def parse_water_delivery(dataset: contraqctor.contract.Dataset) -> pd.Series:
    water_delivery = dataset.at("Behavior").at("HarpBehavior").load().at("OutputSet").load().data.copy()
    water_delivery = water_delivery[(water_delivery["MessageType"] == "WRITE") & (water_delivery["SupplyPort0"])][
        "SupplyPort0"
    ]
    return water_delivery


def parse_odor_onset(dataset: contraqctor.contract.Dataset) -> pd.Series:
    odor_onset = dataset.at("Behavior").at("HarpOlfactometer").load().at("EndValveState").load().data
    odor_onset = odor_onset[odor_onset["MessageType"] == "WRITE"]["EndValve0"]
    odor_onset = odor_onset[(odor_onset) & (~odor_onset.shift(1, fill_value=False))]
    return odor_onset


def parse_continuous_patch_state(dataset: contraqctor.contract.Dataset) -> pd.DataFrame:
    patches_state = dataset.at("Behavior").at("SoftwareEvents").at("PatchState").load().data.copy()
    expanded = pd.json_normalize(patches_state["data"])
    expanded.index = patches_state.index
    patches_state = patches_state.join(expanded)
    return patches_state


def parse_patch_state_at_reward(dataset: contraqctor.contract.Dataset) -> pd.DataFrame:
    patches_state_at_reward = dataset.at("Behavior").at("SoftwareEvents").at("PatchStateAtReward").load().data.copy()
    expanded = pd.json_normalize(patches_state_at_reward["data"])
    expanded.index = patches_state_at_reward.index
    patches_state_at_reward = patches_state_at_reward.join(expanded)
    return patches_state_at_reward


def parse_reward_metadata(dataset: contraqctor.contract.Dataset) -> pd.DataFrame:
    reward_metadata = dataset.at("Behavior").at("SoftwareEvents").at("GiveReward").load().data.copy()
    return reward_metadata

_TSliceable = t.TypeVar("_TSliceable", pd.DataFrame, pd.Series)

def slice_by_index(df: _TSliceable, start_time: float, end_time: float) -> _TSliceable:
    """
    Subsets the DataFrame to only include rows within the specified range.
    Assumes the DataFrame index is a datetime-like index.
    """
    return df[(df.index >= start_time) & (df.index < end_time)]


def process_sites(dataset: contraqctor.contract.Dataset) -> list[Site]:
    """
    Processes sites, patches, and blocks from the dataset and merges them.
    Returns a DataFrame with merged information.
    """
    odor_sites = t.cast(pd.DataFrame, dataset.at("Behavior").at("SoftwareEvents").at("ActiveSite").load().data.copy())
    patches = t.cast(pd.DataFrame, dataset.at("Behavior").at("SoftwareEvents").at("ActivePatch").load().data.copy())
    patches["patch_count"] = range(len(patches))
    blocks = t.cast(pd.DataFrame, dataset.at("Behavior").at("SoftwareEvents").at("Block").load().data.copy())
    blocks["block_count"] = range(len(blocks))

    # Merge nearest patch (backward in time)
    merged = pd.merge_asof(
        odor_sites.sort_index(),
        patches[["data", "patch_count"]].rename(columns={"data": "patch_data"}).sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
        suffixes=("", "_patch"),
    )
    merged["patch_index"] = merged["patch_data"].apply(lambda d: d["state_index"])

    # Merge nearest block (backward in time)
    merged = pd.merge_asof(
        merged.sort_index(),
        blocks[["block_count"]].sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )

    choice_feedback = parse_speaker_choice_feedback(dataset)
    water_delivery = parse_water_delivery(dataset)
    reward_metadata = parse_reward_metadata(dataset)
    odor_onset = parse_odor_onset(dataset)
    continuous_patch_state = parse_continuous_patch_state(dataset)
    patch_state_at_reward = parse_patch_state_at_reward(dataset)

    ## ongoing variables
    current_friction = 0  # TODO

    odor_labels = ""  # TODO

    current_block_idx = 0
    current_patch_idx = 0
    current_patch_in_block_idx = 0
    current_site_in_patch_idx = 0
    current_site_in_block_idx = 0

    sites: list[Site] = []
    for i in range(len(merged) - 1):
        this_timestamp = merged.index[i]
        next_timestamp = merged.index[i + 1] if i < len(merged) - 1 else merged.index[i]

        this_site = vrf_task.VirtualSite(**merged.iloc[i]["data"])
        this_patch = vrf_task.Patch(**merged.iloc[i]["patch_data"])

        site_choice_feedback = slice_by_index(choice_feedback, this_timestamp, next_timestamp)
        assert len(site_choice_feedback) <= 1, "Multiple speaker choices in site interval"

        site_water_delivery = slice_by_index(water_delivery, this_timestamp, next_timestamp)
        assert len(site_water_delivery) <= 1, "Multiple water deliveries in site interval"

        site_odor_onset = slice_by_index(odor_onset, this_timestamp, next_timestamp)
        assert len(site_odor_onset) <= 1, "Multiple odor onsets in site interval"

        site_continuous_patch_state = slice_by_index(continuous_patch_state, this_timestamp, next_timestamp).where(
            continuous_patch_state["PatchId"] == merged.iloc[i]["patch_index"]
        )

        site_patch_state_at_reward = slice_by_index(patch_state_at_reward, this_timestamp, next_timestamp).where(
            patch_state_at_reward["PatchId"] == merged.iloc[i]["patch_index"]
        )
        assert len(site_patch_state_at_reward) <= 1, "Multiple patch states at reward in site interval"

        ##
        this_block_idx = merged.iloc[i]["block_count"]
        this_patch_idx = merged.iloc[i]["patch_count"]

        # We always increment these eagerly
        current_site_in_patch_idx += 1
        current_site_in_block_idx += 1

        # If the patch changed, we reset the site_in_patch counter and increment the patch_in_block counter
        if this_patch_idx != current_patch_idx:
            current_patch_idx = this_patch_idx
            current_site_in_patch_idx = 0
            current_patch_in_block_idx += 1

        # If the blocked changed, we reset both the patch_in_block and site_in_block counters
        # We dont need to re-reset current_patch_idx because patches are unique across blocks
        if this_block_idx != current_block_idx:
            current_block_idx = this_block_idx
            current_patch_in_block_idx = 0
            current_site_in_block_idx = 0

        choice_time = site_choice_feedback.index[0] if not site_choice_feedback.empty else np.nan
        odor_onset_time = site_odor_onset.index[0] if not site_odor_onset.empty else np.nan
        reward_onset_time = site_water_delivery.index[0] if not site_water_delivery.empty else np.nan

        site = Site(
            start_time=this_timestamp,
            stop_time=next_timestamp,
            start_position=this_site.start_position,
            length=this_site.length,
            site_label=str(this_site.label),
            friction=current_friction,
            patch_label=str(this_patch.label),
            odor_label=odor_labels,
            odor_concentration=[this_patch.odor_specification.concentration],  # TODO handle multiple odors
            patch_index=current_patch_idx,
            patch_in_block_index=current_patch_in_block_idx,
            site_index=i,
            site_in_patch_index=current_site_in_patch_idx,
            site_in_block_index=current_site_in_block_idx,
            site_by_type_in_patch_index=0,  # TODO
            odor_onset_time=odor_onset_time,
            reward_onset_time=reward_onset_time,
            odor_offset_time=np.nan,  # TODO
            reward_amount=np.nan if site_patch_state_at_reward.empty else site_patch_state_at_reward.iloc[0]["Amount"],
            reward_probability=np.nan
            if site_patch_state_at_reward.empty
            else site_patch_state_at_reward.iloc[0]["Probability"],
            reward_available=np.nan
            if site_patch_state_at_reward.empty
            else site_patch_state_at_reward.iloc[0]["Available"],
            has_reward=False,  # TODO
            choice_cue_time=choice_time,
            has_choice=not site_choice_feedback.empty,
            has_lick=False,  # TODO
            reward_delay_duration=reward_onset_time - odor_onset_time,
            has_waited_reward_delay=False,  # TODO
            block_index=this_block_idx,
        )
        sites.append(site)
    return sites
