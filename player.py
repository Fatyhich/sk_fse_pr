import cv2
import fire
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Sentinel:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"<{self.name}>"
    
NOT_DEFINED = Sentinel('NOT_DEFINED')

class Plot:
    def __init__(self, root):
        self._image_dir = sorted(Path(root).joinpath('frames').glob('*.jpg'))

        # Open and Read the odometry
        path_to_file = Path(root).joinpath('odometry.csv')
        data = pd.read_csv(path_to_file)

        self._trajectories = data.iloc[:, 4:6].values
        self._X, self._Y = zip(*self._trajectories)

        quaternions = data.iloc[:, :4].values
        self._euler_angles = np.round(np.array([self.quaternion_to_euler(q) for q in quaternions]), 2)

        x_column = data['cart_x']
        y_column = data['cart_y']

        self.np_coord_x = x_column.to_numpy()
        self.np_coord_y = y_column.to_numpy()

        self.fig, self.ax = plt.subplot_mosaic(
            [["frame", "traj"]],
            figsize=(14, 10),
            constrained_layout=True,
        )

        self.fig.suptitle(Path(root).name, fontsize=32)
        # Initialize the FuncAnimation object
        self._animation = animation.FuncAnimation(self.fig, self.animate, frames=len(self), repeat=False)

    def plot_frame(self, index):
        img = cv2.imread(self._image_dir[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame = np.array(img)
        self.ax['frame'].imshow(frame)
        self.ax["frame"].set_title('RGB_Left Frame', fontsize=20)
        self.ax['frame'].axis('off')

    def plot_trajectory(self, index, arrow_length=2.):
        x, y = self._trajectories[index]
        yaw = self._euler_angles[index][2]
        dx = arrow_length * np.cos(yaw)
        dy = arrow_length * np.sin(yaw)

        self.ax["traj"].plot(self._X, self._Y, label='Trajectory', color='blue')
        self.ax["traj"].arrow(x, y, dx, dy, head_width=1., head_length=2., fc='red', ec='red')

        traj_x_range = max(self._X) - min(self._X)
        traj_y_range = max(self._Y) - min(self._Y)
        max_range = max(traj_x_range, traj_y_range)
        traj_x_mid = (max(self._X) + min(self._X)) / 2
        traj_y_mid = (max(self._Y) + min(self._Y)) / 2

        self.ax["traj"].set_xlim((traj_x_mid - max_range / 2) - 1,
                                 (traj_x_mid + max_range / 2) + 1)

        self.ax["traj"].set_ylim((traj_y_mid - max_range / 2) - 1,
                                 (traj_y_mid + max_range / 2) + 1)

        self.ax["traj"].set_xlabel('X Position')
        self.ax["traj"].set_ylabel('Y Position')
        self.ax["traj"].set_title('Trajectory with Yaw Direction', fontsize=20)
        self.ax["traj"].legend()
        self.ax["traj"].set_aspect("equal")

        plt.grid()

    def quaternion_to_euler(self, q):
        # Extracting
        x, y, z, w = q

        # Quaternion to Euler
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(t3, t4)

        return np.array([roll, pitch, yaw])  # Возвращаем углы Эйлера

    def plot(self, index):
        self.plot_frame(index)
        if self._X > 100 and self._Y > 100:
            _arrow_length = 4.
        elif self._X > 500 and self._Y > 500:
            _arrow_length = 6.
        elif self._X > 1000 and self._Y > 1000:
            _arrow_length = 10.
        self.plot_trajectory(index, _arrow_length)
        plt.show()

    def animate(self, index):
        print(f"Animating frame {index}")
        self.ax["frame"].clear()
        self.ax["traj"].clear()

        # Plot frame and trajectory for the current index
        self.plot_frame(index)
        self.plot_trajectory(index)

    def save_video(self, output_filename, fps=60):
        self._animation.save(output_filename, writer='ffmpeg', fps=fps)

    def show(self):
        plt.show()

    def __len__(self):
        return len(self._image_dir)


def main(input_dir: str = NOT_DEFINED, output_file_name: str = 'trajectory'):
    input_dir = Path(input_dir)
    output_file_name = Path(output_file_name)
    
    file_name = f'{output_file_name}.mp4'
    assert input_dir, 'input_dir not defined!'

    assert Path(input_dir).is_dir(), \
        f"--input_dir parameter should be an existing folder but is not : {input_dir} Exit program."
    assert Path(input_dir).joinpath('frames').is_dir(), \
        "--input_dir should contain the subfolder with frames but is not. Exit program."
    assert Path(input_dir).joinpath('frames').glob('*.jpg'), \
        "--input_dir should contain the subfolder with frames but empty. Exit program."
    assert Path(input_dir).glob('odometry.csv'), \
        "--input_dir should contain the file with odometry but is not. Exit program."

    plot = Plot(input_dir)
    plot.save_video(file_name)


if __name__ == "__main__":
    fire.Fire(main)
