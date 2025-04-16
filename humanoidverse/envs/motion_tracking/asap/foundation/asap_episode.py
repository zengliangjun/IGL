
import torch
from humanoidverse.envs.legged_base_task.term.foundation import episode
from loguru import logger

#from humanoidverse.envs.base_task.term import base
#class ASAPEpisode(base.BaseManager):
class ASAPEpisode(episode.LeggedEpisode):
    def __init__(self, _task):
        super(ASAPEpisode, self).__init__(_task)

    def init(self):
        super(ASAPEpisode, self).init()
        self.terminate_when_motion_far_threshold = self.config.termination_scales.termination_motion_far_threshold
        logger.info(f"Terminate when motion far threshold: {self.terminate_when_motion_far_threshold}")

    def _update_reset_buf(self):
        super(ASAPEpisode, self)._update_reset_buf()

        if not self.config.termination.terminate_when_motion_far:
            return

        if not hasattr(self.task, "robotstatus_manager"):
            return

        robotstatus_manager = self.task.robotstatus_manager

        ref_body_pos = robotstatus_manager.motion_res["rg_pos_t"]
        ## diff compute - kinematic position
        _dif_global_body_pos = ref_body_pos - robotstatus_manager._rigid_body_pos

        reset_buf_motion_far = torch.any(torch.norm(_dif_global_body_pos, dim=-1) > self.terminate_when_motion_far_threshold, dim=-1)
        self.reset_buf |= reset_buf_motion_far
        # log current motion far threshold

    def _update_timeout_buf(self):
        super(ASAPEpisode, self)._update_timeout_buf()

        if self.config.termination.terminate_when_motion_end:
            robotdata_manager = self.task.robotdata_manager
            current_time = (self.episode_length_buf) * self.task.dt + robotdata_manager.motion_start_times
            self.time_out_buf |= current_time > robotdata_manager.motion_len
