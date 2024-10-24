import threading
import fire
import tkinter as tk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
from canguro_processing_tools.utils.io_utils import AbstractImageIO, OpenCVImageIO, TurboJPEGImageIO, ImageIO


class Trajectory:

    def __init__(self, trajectory_path: str | Path) -> None:
        trajectory_path = Path(trajectory_path)
        odometry = self._read_odometry(trajectory_path / "odometry.csv")
        frames = self._locate_frames(trajectory_path / "frames")
        assert len(frames) == odometry.shape[0], f"Number of frames does not match odometry length!"
        self._odometry = odometry
        self._frames = frames

    @property
    def poses(self) -> np.ndarray:
        return self._odometry

    @property
    def frames(self) -> list[Path]:
        return self._frames

    def __len__(self) -> int:
        return self._odometry.shape[0]

    def _read_odometry(self, csv_path: Path) -> np.ndarray:
        data = pd.read_csv(str(csv_path))
        x = data["cart_x"].to_numpy()
        y = data["cart_y"].to_numpy()
        yaw = np.array([self._quaternion_to_yaw(data["quat_x"][i],
                                                data["quat_y"][i],
                                                data["quat_z"][i],
                                                data["quat_w"][i]) for i in range(x.shape[0])])
        result = np.column_stack((x, y, yaw))
        return result

    def _locate_frames(self, frames_dir: Path) -> list[Path]:
        return sorted(frames_dir.glob("*.jpg"))

    def _quaternion_to_yaw(self, x: float, y: float, z: float, w: float) -> float:
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)
        return yaw


class App(tk.Tk):
    def __init__(self, trajectory, image_loader: AbstractImageIO):
        super(App, self).__init__()
        self._trajectory = trajectory
        self._image_loader = image_loader
        self._num_frames = len(trajectory)
        self._current_frame = 0
        self._auto_play = False

        self.title("Trajectory Viewer")
        self.geometry("800x600")

        self._fig, self._ax = plt.subplots()
        self._ax.plot(self._trajectory.poses[:, 0], self._trajectory.poses[:, 1], 'b-')

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
        img_path = str(self._trajectory.frames[frame_idx])
        img = self._image_loader.read(img_path)
        img = Image.fromarray(img)
        img = img.resize((400, 300))
        photo = ImageTk.PhotoImage(img)
        self._img_label.config(image=photo)
        self._img_label.image = photo

    def _plot_trajectory(self, frame_idx):
        self._ax.clear()
        self._ax.plot(self._trajectory.poses[:, 0], self._trajectory.poses[:, 1], 'b-')
        x, y, yaw = self._trajectory.poses[frame_idx]
        self._ax.plot(x, y, 'ro')
        self._ax.arrow(x, y, np.cos(yaw), np.sin(yaw), color='r', head_width=0.1)
        self._canvas.draw()

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


def main(src: str,
         image_io: str = "turbo"):
    image_io = ImageIO(image_io)
    trajectory = Trajectory(src)
    if image_io == ImageIO.OPENCV:
        loader = OpenCVImageIO(read_bgr=False, write_bgr=False)
    elif image_io == ImageIO.TURBO_JPEG:
        loader = TurboJPEGImageIO(read_format="rgb", write_format="rgb")
    else:
        raise ValueError(f"Unknown image IO type {image_io}")

    def run_app():
        app = App(trajectory, loader)
        app.mainloop()
    
    app_thread = threading.Thread(target=run_app)
    app_thread.start()
    app_thread.join()  # Wait for the thread to finish


if __name__ == "__main__":
    fire.Fire(main)
