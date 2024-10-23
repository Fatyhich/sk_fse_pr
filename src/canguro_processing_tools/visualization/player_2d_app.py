import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt


from typing import Union
from pathlib import Path
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
from canguro_processing_tools.trajectories.trajectory_2d import Trajectory2D
from canguro_processing_tools.utils.io_utils import (ImageIO,
                                                     OpenCVImageIO,
                                                     TurboJPEGImageIO)


class Trajectory2dVisualApp(tk.Tk):
    def __init__(self, 
                 extraction_path: Union[str, Path], 
                 image_io: ImageIO):
        super(Trajectory2dVisualApp, self).__init__()
        if image_io == ImageIO.OPENCV:
            loader = OpenCVImageIO(False, False)
        elif image_io == ImageIO.TURBO_JPEG:
            loader = TurboJPEGImageIO("rgb", "rgb")
        else:
            raise ValueError(f"Unknown image IO method {image_io}")

        initial_trajectory = Trajectory2D.read(Path(extraction_path))
        if not initial_trajectory.is_complete:
            print(f"Warning: trajectory is not complete {len(initial_trajectory.poses_timestamps)} / {len(initial_trajectory.frames_timestamps)} available")
        complete_trajectories = initial_trajectory.split_on_completes()

        all_timestamps = initial_trajectory.frames_timestamps
        complete_chunk_bounds = []
        complete_timestamps = set()
        for complete_trajectory in complete_trajectories:
            timestamps = complete_trajectory.frames_timestamps
            complete_timestamps = complete_timestamps.union(set(timestamps))
            complete_chunk_bounds.append((timestamps[0], timestamps[-1]))

        self._frames = initial_trajectory.frames_left
        self._trajectory = initial_trajectory
        self._complete_trajectories = complete_trajectories
        self._all_timestamps = initial_trajectory.frames_timestamps
        self._complete_timestamps = complete_timestamps
        self._complete_chunk_bounds = complete_chunk_bounds
        self._all_poses = initial_trajectory.poses
        self._image_loader = loader
        self._num_frames = len(all_timestamps)
        self._current_frame = 0
        self._auto_play = False

        self.title("Trajectory Viewer")
        self.geometry("800x600")

        self._fig, self._ax = plt.subplots()
        self._plot_trajectory_chunks()

        self._canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self._img_label = tk.Label(self)
        self._img_label.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self._slider = ttk.Scale(self, from_=0, to=self._num_frames - 1, orient="horizontal", command=self._update_frame)
        self._slider.pack(side=tk.BOTTOM, fill=tk.X)

        self._auto_play_button = tk.Button(self, text="Start Auto Play", command=self._toggle_auto_play)
        self._auto_play_button.pack(side=tk.BOTTOM)

        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        self._update_frame(0)

    def _update_frame(self, value):
        frame_idx = int(float(value))
        self._current_frame = frame_idx

        self._display_image(frame_idx)

        self._plot_trajectory(frame_idx)

    def _display_image(self, frame_idx):
        img_path = str(self._frames[frame_idx])
        img = self._image_loader.read(img_path)
        img = Image.fromarray(img)
        img = img.resize((400, 300))
        photo = ImageTk.PhotoImage(img)
        self._img_label.config(image=photo)
        self._img_label.image = photo

    def _plot_trajectory_chunks(self):
        for trajectory in self._complete_trajectories:
            poses = trajectory.poses
            self._ax.plot(poses[:, 0], poses[:, 1], 'b-')

    def _plot_trajectory(self, frame_idx):
        self._ax.clear()
        self._plot_trajectory_chunks()

        current_timestamp = self._all_timestamps[frame_idx]
        if current_timestamp in self._complete_timestamps:
            x, y, yaw = self._trajectory.pose_at(current_timestamp)
            self._ax.plot(x, y, 'ro')
            self._ax.arrow(x, y, np.cos(yaw), np.sin(yaw), color='r', head_width=0.1)
        
        self._canvas.draw()
        # else:
        #     if current_timestamp < self._complete_chunk_bounds[0][0]:
        #         pseudo_timestamp = self._complete_chunk_bounds[0][0]
        #     else:
        #         chunk_idx = len(self._complete_chunk_bounds) - 1
        #         for i, bounds in self._complete_chunk_bounds:
        #             if current_timestamp < bounds[1]:
        #                 chunk_idx = i - 1
        #                 break
        #         pseudo_timestamp = self._complete_chunk_bounds[chunk_idx][1]
        #     x, y, yaw = self._trajectory.pose_at(pseudo_timestamp)
        #     self._ax.plot(x, y, 'ro')
        #     # self._ax.arrow(x, y, np.cos(yaw), np.sin(yaw), color='r', head_width=0.1)
        #     self._canvas.draw()

    def _toggle_auto_play(self):
        self._auto_play = not self._auto_play
        if self._auto_play:
            self._auto_play_button.config(text="Stop Auto Play")
            self._auto_play_frames()
        else:
            self._auto_play_button.config(text="Start Auto Play")

    def _auto_play_frames(self):
        if self._auto_play:
            next_frame = (self._current_frame + 1) % self._num_frames
            self._slider.set(next_frame)
            self.after(1, self._auto_play_frames)  # Adjust the interval (in milliseconds) as desired

    def _on_closing(self):
        self._auto_play = False
        self.destroy()
