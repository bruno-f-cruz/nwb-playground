import logging
import typing as t

import aind_behavior_vr_foraging
import contraqctor
import numpy as np
import pandas as pd
import semver
from contraqctor.contract.json import PydanticModel
from pydantic import BaseModel

from .models import Site

logger = logging.getLogger(__name__)

_TSliceable = t.TypeVar("_TSliceable", pd.DataFrame, pd.Series)


class DatasetProcessor:
    def __init__(self, dataset: contraqctor.contract.Dataset) -> None:
        self.dataset = dataset

        if self.dataset_version != self.parser_version:
            logger.warning(
                "Dataset version %s does not match parser version %s", self.dataset_version, self.parser_version
            )

    @property
    def dataset_version(self) -> semver.Version:
        return self._parse_version(self.dataset.version)

    @property
    def parser_version(self) -> semver.Version:
        return semver.Version.parse(aind_behavior_vr_foraging.__semver__)

    @staticmethod
    def _parse_version(value: str | semver.Version) -> semver.Version:
        if isinstance(value, semver.Version):
            return value
        return semver.Version.parse(value)

    @staticmethod
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

    @staticmethod
    def _parse_speaker_choice_feedback(dataset: contraqctor.contract.Dataset) -> pd.DataFrame:
        speaker_choice = dataset.at("Behavior").at("HarpBehavior").load().at("PwmStart").load().data
        speaker_choice = speaker_choice[(speaker_choice["MessageType"] == "WRITE") & (speaker_choice["PwmDO2"])]
        return speaker_choice

    @staticmethod
    def _parse_water_delivery(dataset: contraqctor.contract.Dataset) -> pd.Series:
        water_delivery = dataset.at("Behavior").at("HarpBehavior").load().at("OutputSet").load().data
        water_delivery = water_delivery[(water_delivery["MessageType"] == "WRITE") & (water_delivery["SupplyPort0"])][
            "SupplyPort0"
        ]
        return water_delivery

    @staticmethod
    def _parse_odor_onset(dataset: contraqctor.contract.Dataset) -> pd.Series:
        odor_onset = dataset.at("Behavior").at("HarpOlfactometer").load().at("EndValveState").load().data
        odor_onset = odor_onset[odor_onset["MessageType"] == "WRITE"]["EndValve0"]
        odor_onset = odor_onset[(odor_onset) & (~odor_onset.shift(1, fill_value=False))]
        return odor_onset

    @staticmethod
    def _parse_continuous_patch_state(dataset: contraqctor.contract.Dataset) -> pd.DataFrame:
        patches_state = dataset.at("Behavior").at("SoftwareEvents").at("PatchState").load().data
        expanded = pd.json_normalize(patches_state["data"])
        expanded.index = patches_state.index
        patches_state = patches_state.join(expanded)
        return patches_state

    @staticmethod
    def _parse_patch_state_at_reward(dataset: contraqctor.contract.Dataset) -> pd.DataFrame:
        patches_state_at_reward = dataset.at("Behavior").at("SoftwareEvents").at("PatchStateAtReward").load().data
        expanded = pd.json_normalize(patches_state_at_reward["data"])
        expanded.index = patches_state_at_reward.index
        patches_state_at_reward = patches_state_at_reward.join(expanded)
        return patches_state_at_reward

    @staticmethod
    def _parse_wait_reward_outcome(dataset: contraqctor.contract.Dataset) -> pd.Series:
        try:
            return dataset.at("Behavior").at("SoftwareEvents").at("WaitRewardOutcome").load().data
        except FileNotFoundError:
            return pd.Series(dtype=bool)

    @staticmethod
    def _parse_reward_metadata(dataset: contraqctor.contract.Dataset) -> pd.DataFrame:
        reward_metadata = dataset.at("Behavior").at("SoftwareEvents").at("GiveReward").load().data
        return reward_metadata

    @staticmethod
    def _as_dict(d: contraqctor.contract.DataStream | PydanticModel | BaseModel | dict) -> dict:
        if isinstance(d, (PydanticModel, contraqctor.contract.DataStream)):
            d = d.data
        if isinstance(d, dict):
            return d
        if isinstance(d, BaseModel):
            return d.model_dump()
        else:
            raise TypeError(f"Cannot convert type {type(d)} to dict")

    @staticmethod
    def _parse_friction(dataset: contraqctor.contract.Dataset) -> pd.DataFrame:
        d = dataset.at("Behavior").at("HarpTreadmill").at("BrakeCurrentSetPoint").load().data
        return d.loc[d["MessageType"] == "WRITE", "BrakeCurrentSetPoint"]

    @staticmethod
    def slice_by_index(df: _TSliceable, start_time: float, end_time: float) -> _TSliceable:
        """
        Subsets the DataFrame to only include rows within the specified range.
        Assumes the DataFrame index is a datetime-like index.
        """
        return df[(df.index >= start_time) & (df.index < end_time)]

    def get_olfactometer_channel_count(self, dataset: contraqctor.contract.Dataset) -> int:
        if self.dataset_version < semver.Version.parse("0.7.0"):
            return 3  # The channel 3 is always used as carrier, therefor only 3 odor channels are available.
        else:
            raise NotImplementedError("Olfactometer channel count parsing not implemented for rig versions > 0.6.4")

    def process_odor_concentration(self, odor_specification: BaseModel | dict | None, n_channels: int) -> list[float]:
        concentration = [0.0] * n_channels
        if odor_specification is None:
            return concentration
        if isinstance(odor_specification, BaseModel):
            odor_specification = odor_specification.model_dump()

        match v := self.dataset_version:
            case _ if v < semver.Version.parse("0.7.0"):
                index = odor_specification.get("index")
                if not isinstance(index, int):
                    raise TypeError("odor_specification.index must be an int")
                concentration[index] = odor_specification.get("concentration", 0.0)
            case _:
                raise NotImplementedError("OdorSpecification processing not implemented for rig versions >= 0.7.0")
        return concentration

    def process(self) -> list[Site]:
        """
        Processes sites, patches, and blocks from the dataset and merges them.
        Returns a DataFrame with merged information.
        """
        dataset = self.dataset
        odor_sites = t.cast(pd.DataFrame, dataset.at("Behavior").at("SoftwareEvents").at("ActiveSite").load().data)
        patches = t.cast(pd.DataFrame, dataset.at("Behavior").at("SoftwareEvents").at("ActivePatch").load().data)
        patches["patch_count"] = range(len(patches))
        blocks = t.cast(pd.DataFrame, dataset.at("Behavior").at("SoftwareEvents").at("Block").load().data)
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

        choice_feedback = self._parse_speaker_choice_feedback(dataset)
        water_delivery = self._parse_water_delivery(dataset)
        reward_metadata = self._parse_reward_metadata(dataset)
        odor_onset = self._parse_odor_onset(dataset)
        continuous_patch_state = self._parse_continuous_patch_state(dataset)
        patch_state_at_reward = self._parse_patch_state_at_reward(dataset)
        friction = self._parse_friction(dataset)
        olfactometer_channel_count = self.get_olfactometer_channel_count(dataset)
        wait_reward_outcome = self._parse_wait_reward_outcome(dataset)

        ## ongoing variables
        current_friction = 0

        current_block_idx = 0
        current_patch_idx = 0
        current_patch_in_block_idx = 0
        current_site_in_patch_idx = 0
        current_site_in_block_idx = 0

        sites: list[Site] = []
        for i in range(len(merged) - 1):
            this_timestamp = merged.index[i]
            next_timestamp = merged.index[i + 1] if i < len(merged) - 1 else merged.index[i]

            this_site = merged.iloc[i]["data"]
            this_patch = merged.iloc[i]["patch_data"]

            site_choice_feedback = self.slice_by_index(choice_feedback, this_timestamp, next_timestamp)
            assert len(site_choice_feedback) <= 1, "Multiple speaker choices in site interval"

            site_water_delivery = self.slice_by_index(water_delivery, this_timestamp, next_timestamp)
            assert len(site_water_delivery) <= 1, "Multiple water deliveries in site interval"

            site_odor_onset = self.slice_by_index(odor_onset, this_timestamp, next_timestamp)
            assert len(site_odor_onset) <= 1, "Multiple odor onsets in site interval"

            this_friction = self.slice_by_index(friction, this_timestamp, next_timestamp)
            if not this_friction.empty:
                current_friction = this_friction.values[-1]

            site_continuous_patch_state = self.slice_by_index(continuous_patch_state, this_timestamp, next_timestamp)
            site_continuous_patch_state = site_continuous_patch_state[
                site_continuous_patch_state["PatchId"] == merged.iloc[i]["patch_index"]
            ]

            site_patch_state_at_reward = self.slice_by_index(patch_state_at_reward, this_timestamp, next_timestamp)
            site_patch_state_at_reward = site_patch_state_at_reward[
                site_patch_state_at_reward["PatchId"] == merged.iloc[i]["patch_index"]
            ]
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
            reward_metadata_sliced = self.slice_by_index(reward_metadata, this_timestamp, next_timestamp)
            if reward_metadata_sliced.empty:
                reward_onset_time = np.nan  # Just in case there is some manual water delivery we exclude it here
            else:
                if len(site_water_delivery) == 0:
                    logger.warning("Reward metadata found but no water delivery in site interval")
                    reward_onset_time = np.nan
                elif len(reward_metadata_sliced) > 1:
                    logger.warning("Multiple reward metadata entries in site interval...Using first one")
                    reward_onset_time = site_water_delivery.index[0]
                else:
                    reward_onset_time = site_water_delivery.index[0] if not site_water_delivery.empty else np.nan

            wait_reward_outcome_sliced = self.slice_by_index(wait_reward_outcome, this_timestamp, next_timestamp)
            has_waited_reward_delay: t.Optional[bool]
            if wait_reward_outcome_sliced.empty:
                has_waited_reward_delay = None
            else:
                if len(wait_reward_outcome_sliced) > 1:
                    logger.warning("Multiple wait reward outcome entries in site interval...Using first one")
                has_waited_reward_delay = wait_reward_outcome_sliced.iloc[0]["data"]["IsSuccessfulWait"]

            site = Site(
                start_time=this_timestamp,
                stop_time=next_timestamp,
                start_position=this_site["start_position"],
                length=this_site["length"],
                site_label=str(this_site["label"]),
                friction=current_friction,
                patch_label=str(this_patch["label"]),
                odor_concentration=self.process_odor_concentration(
                    this_patch["odor_specification"], olfactometer_channel_count
                ),
                patch_index=current_patch_idx,
                patch_in_block_index=current_patch_in_block_idx,
                site_index=i,
                site_in_patch_index=current_site_in_patch_idx,
                site_in_block_index=current_site_in_block_idx,
                site_by_type_in_patch_index=0,  # TODO
                odor_onset_time=odor_onset_time,
                reward_onset_time=reward_onset_time,
                reward_amount=np.nan
                if site_patch_state_at_reward.empty
                else site_patch_state_at_reward.iloc[0]["Amount"],
                reward_probability=np.nan
                if site_patch_state_at_reward.empty
                else site_patch_state_at_reward.iloc[0]["Probability"],
                reward_available=np.nan
                if site_patch_state_at_reward.empty
                else site_patch_state_at_reward.iloc[0]["Available"],
                has_reward=np.isnan(reward_onset_time) == False,
                choice_cue_time=choice_time,
                has_choice=not site_choice_feedback.empty,
                reward_delay_duration=reward_onset_time - odor_onset_time,
                has_waited_reward_delay=has_waited_reward_delay,
                block_index=this_block_idx,
            )
            sites.append(site)
        return sites
