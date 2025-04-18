
    ## help function
    def next_task(self):
        # This function is only called when evaluating
        self.motion_start_idx += self.num_envs
        if self.motion_start_idx >= self.num_motions:
            self.motion_start_idx = 0

        if self.task.is_evaluating:
            self._motion_lib.load_motions(random_sample=False, start_idx=self.motion_start_idx)
        else:
            self._motion_lib.load_motions(random_sample=True, start_idx=self.motion_start_idx)

    # only for set_is_evaluating random_sample is false
    def with_evaluating(self):
        if not self.task.is_evaluating:
            return

        logger.info(f"reset with evaluating model with self._motion_lib.load_motions(random_sample=False)")
        self._motion_lib.load_motions(random_sample=False)


    def _pre_play(self):
        if not hasattr(self.task, 'is_motion_player') or not self.task.is_motion_player:
            return
        ## only for player
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self.task.need_to_refresh_envs[env_ids] = True




## only for training with random_sample update
        if not self.config.resample_motion_when_training:
            return

        if not hasattr(self.task, 'episode_manager'):
            return

        episode_manager = self.task.episode_manager
        if episode_manager.common_step_counter.item() % self.resample_time_interval:
            return

        if 0 == episode_manager.common_step_counter.item():
            return

        logger.info(f"Resampling motion at step {episode_manager.common_step_counter.item()}")
        self._motion_lib.load_motions(random_sample=True)
        episode_manager.reset_buf[:] = 1