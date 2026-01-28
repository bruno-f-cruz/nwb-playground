import dataclasses
from typing import List, Optional

from pydantic import BaseModel, Field


@dataclasses.dataclass
class Trial:
    odor_onset_time: float
    choice_time: Optional[float]
    reward_time: Optional[float]
    reaction_duration: Optional[float]
    patch_index: int
    is_rewarded: Optional[bool]
    is_choice: bool
    p_reward: float
    begin_stop_time: Optional[float]
    longest_stop_duration: Optional[float]
    start_time: float
    stop_time: float


class Site(BaseModel):
    """A model representing a virtual site in the VR foraging task."""

    start_time: float = Field(description="Start time, in software, for this site. (unit: second)")
    stop_time: float = Field(description="Stop time, in software, for this site. (unit: second)")
    start_position: float = Field(
        description="Start coordinate for this site in the VR environment. (unit: centimeter)"
    )
    length: float = Field(description="The length of the site. (unit: centimeter)")
    site_label: str = Field(description="Label of the site")
    friction: float = Field(description="Assigned friction for the site. (unit: percentage)")
    patch_label: str = Field(description="Patch type name")
    odor_label: str = Field(description="Odor molecule assigned to patch")
    odor_concentration: List[float] = Field(
        description="An array representing the concentration levels of each odor channels. (unit: percentage)"
    )
    patch_index: int = Field(description="Patch number within the session")
    patch_in_block_index: int = Field(description="Patch number within the block")
    site_index: int = Field(description="Site number within the session")
    site_in_block_index: int = Field(description="Site number within the block")
    site_in_patch_index: int = Field(description="Site number within the patch")
    site_by_type_in_patch_index: int = Field(
        description="Same as site_in_patch_index but only counting sites of the same type (e.g. RewardSite)"
    )
    odor_onset_time: Optional[float] = Field(
        None, description="Time of odor onset. Will be null if no odor was delivered. (unit: second)"
    )
    odor_offset_time: Optional[float] = Field(
        None, description="Time of odor offset. Will be null if no odor was delivered. (unit: second)"
    )
    reward_onset_time: Optional[float] = Field(None, description="Time when reward was delivered. (unit: second)")
    reward_amount: Optional[float] = Field(None, description="Amount of reward delivered. (unit: milliliter)")
    reward_probability: Optional[float] = Field(
        None,
        description="Reward probability at the time of the reward delivery. Will be null if the reward is not sampled (e.g. has_choice is False). (unit: percentage)",
    )
    reward_available: Optional[float] = Field(
        None,
        description="Reward left at the time of reward delivery. Will be null if the reward is not sampled (e.g. has_choice is False). (unit: milliliter)",
    )
    has_reward: Optional[bool] = Field(None, description="Boolean whether reward was delivered, bool.")
    choice_cue_time: Optional[float] = Field(
        None,
        description="Time when choice cue was delivered. Also can be considered the stop cue. The choice tone is delivered when a stop is successful. (unit: second)",
    )
    has_choice: Optional[bool] = Field(None, description="Defines whether a choice occurred in the site.")
    has_lick: bool = Field(description="Defines whether a lick occurred in the site.")
    reward_delay_duration: Optional[float] = Field(
        None, description="reward_onset_time - choice_cue_time. (unit: second)"
    )
    has_waited_reward_delay: Optional[bool] = Field(
        None,
        description="Boolean whether the mouse successfully waited through the reward delay to get the reward. Will be null if has_choice is false.",
    )
    block_index: int = Field(description="Block number within the session")
