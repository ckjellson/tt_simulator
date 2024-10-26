import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mpl_toolkits.axes_grid1
import matplotlib.widgets
from argparse import ArgumentParser

# Physics parameters
xtable = 2.74
ytable = 1.525
neth = 0.1525
ball_radius = 0.019
cor = 0.9  # Standard coefficient of restitution for a table tennis ball
gravity = np.array([0, 0, -9.81])
air_density = 1.225
ball_mass = 2.7 / 1000
drag_coeff = 0.47  # Approximate drag coefficient for a table tennis ball (in reality needs to be modelled)
air_viscosity = 1.86e-5
table_friction = 0.05  # Made up, varies between tables
apply_slip = True

# Program settings
dt = 0.001
interval = 20
margin = 0.5
max_iterations = 10000
xrange = (-margin, xtable + margin)
rpsmax = 100
trail_length = 20

"""
Table utility functions
"""


def inside_table(x, y):
    return x >= 0 and x <= xtable and y >= 0 and y <= ytable


def inside_table_x(x, y):
    return x >= -margin and x <= xtable + margin


def inside_net(x, y, z):
    return (
        np.abs(x - xtable / 2) < ball_radius
        and np.abs(y - ytable / 2) < ytable / 2 + neth + ball_radius
        and (z > -ball_radius and z < neth + ball_radius)
    )


def plot_table(ax):
    x_ = np.array([0.0, xtable])
    y_ = np.array([0.0, ytable])
    X_, Y_ = np.meshgrid(x_, y_)
    Z_ = X_ * 0
    ax.plot_surface(X_, Y_, Z_, color="b")
    y_ = np.array([0.0 - neth, ytable + neth])
    z_ = np.array([0.0, neth])
    Y_, Z_ = np.meshgrid(y_, z_)
    X_ = np.ones(Y_.shape) * xtable / 2
    ax.plot_surface(X_, Y_, Z_, color="grey", alpha=0.7)


"""
Acceleration computations
"""


def magnus_acc(speed, rotation):
    """
    This function assumes no slip, leading to unrealistic behaviour at high spins.
    """
    velocity = np.linalg.norm(speed)
    proj_rot = np.dot(speed, rotation) / np.dot(speed, speed) * speed
    rot_perpendicular = rotation - proj_rot
    rot_perpendicular_magnitude = np.linalg.norm(rot_perpendicular)
    force = (
        2
        * np.pi
        * rot_perpendicular_magnitude
        * ball_radius**2
        * velocity
        * air_density
        * ball_radius
    )
    if apply_slip:
        # Made up way to model artificial slip:
        force = force / (1 + np.abs(velocity - np.linalg.norm(rotation) * ball_radius))
    force_dir = np.cross(speed, rot_perpendicular)
    if np.linalg.norm(force_dir) == 0:
        return np.array([0, 0, 0])
    force_dir = force_dir / np.linalg.norm(force_dir)
    return force / ball_mass * force_dir


def drag_acc(speed):
    drag_force = (
        air_density
        / 2
        * ball_radius**2
        * np.pi
        * drag_coeff
        * np.linalg.norm(speed) ** 2
    )
    return -drag_force * speed / np.linalg.norm(speed) / ball_mass


def bounce_acc(rotation):
    """
    This function assumes no slipping, leading to unrealistic behaviour at high spins.
    """
    rot_magnitude = np.linalg.norm(rotation)
    normal = np.array([0, 0, 1])
    proj_rot = np.dot(normal, rotation) / np.dot(normal, normal) * normal
    rot_perpendicular = rotation - proj_rot
    if np.linalg.norm(rot_perpendicular) == 0:
        return np.array([0, 0, 0]), rotation
    force_dir = np.cross(normal, rot_perpendicular)
    force_dir = force_dir / np.linalg.norm(force_dir)
    if apply_slip:
        # Made up way to model artificial slip:
        force_magn = (
            rot_magnitude
            * table_friction
            / (np.sqrt(np.max([1, (rot_magnitude - 20 * np.pi) / (2 * np.pi)])))
        )
    else:
        force_magn = rot_magnitude * table_friction
    new_rot = (
        rotation
        - rot_perpendicular
        / np.linalg.norm(rot_perpendicular)
        * force_magn
        / ball_mass
        * ball_radius
    )
    return force_magn * force_dir / ball_mass, new_rot


def viscous_ang_acceleration(rot):
    rot_magnitude = np.linalg.norm(rot)
    if rot_magnitude == 0:
        return rot * 0
    torque = 8 * np.pi * ball_radius**3 * air_viscosity * rot_magnitude
    return -torque * rot / rot_magnitude / ball_mass


"""Creating the ball path"""


def predict(X, dt):
    did_bounce = False
    if -ball_radius * 2 < X[2] + dt * X[5] - ball_radius < 0 and inside_table(
        X[0], X[1]
    ):
        X[5] = -X[5] * cor
        did_bounce = True
    if inside_net(X[0], X[1], X[2]):
        X[3:5] = 0
    X[:3] = X[:3] + dt * X[3:6]
    acceleration = np.copy(gravity)
    acceleration += magnus_acc(X[3:6], X[6:9])
    acceleration += drag_acc(X[3:6])
    if did_bounce:
        bounce_acceleration, new_rot = bounce_acc(X[6:9])
        X[6:9] = new_rot
        acceleration += bounce_acceleration
    ang_acceleration = viscous_ang_acceleration(X[6:9])
    X[3:6] = X[3:6] + dt * acceleration
    X[6:9] = X[6:9] + dt * ang_acceleration
    return X


def generate_path(X):
    ballpath = [np.copy(X)]
    iterations = 0
    while inside_table_x(X[0], X[1]) and X[2] > -ball_radius * 2:
        X_old = np.copy(X)
        X = predict(X, dt)
        if np.abs(X_old[0] - X[0]) < 1e-9 and np.abs(X_old[1] - X[1]) < 1e-9:
            break
        if iterations % interval == 0:
            ballpath.append(np.copy(X))
        iterations += 1
        if iterations > max_iterations:
            break
    positions = [
        [x[0] for x in ballpath],
        [x[1] for x in ballpath],
        [x[2] for x in ballpath],
    ]
    return ballpath, positions


"""
Initial state and path.
The initial state rotation values are given in rad/s instead of rps.
Enter your own initial state here to create a gif of the ball path.
"""
initial_state = np.array(
    [0.0, 0.41758614, 0.305, 2.71260997, 0.0, 0.76246334, 12.16100382, 55.29, 0.0]
)
_, positions = generate_path(np.copy(initial_state))


class Player(FuncAnimation):
    """Class for creating the interactive animation"""

    def __init__(
        self,
        fig,
        ax,
        func,
        frames=None,
        **kwargs,
    ):
        global positions
        self.i = 0
        self.min = 0
        self.max = len(positions[0])
        self.runs = True
        self.fig = fig
        self.ax = ax
        self.func = func
        self.setup()
        if frames is None:
            frames = self.play()
        FuncAnimation.__init__(
            self,
            self.fig,
            self.update,
            frames=frames,
            init_func=None,
            interval=dt * interval * 1000 - 10,
            blit=True,
            **kwargs,
        )

    def update(self, i):
        return self.func(i)

    def play(self):
        while self.runs:
            self.i = self.i + 1
            if self.i > self.min and self.i + 1 < self.max:
                yield self.i
            else:
                self.i = 0
                yield self.i

    def start(self):
        self.runs = True
        self.event_source.start()

    def stop(self, event=None):
        self.runs = False
        self.event_source.stop()

    def forward(self, event=None):
        self.start()

    def update_state(self, val, stateind=None):
        global initial_state
        if stateind > 5:
            initial_state[stateind] = val * 2 * np.pi
        else:
            initial_state[stateind] = val

    def regenerate(self, event=None):
        global positions, initial_state
        _, positions = generate_path(np.copy(initial_state))
        self.max = len(positions[0])
        self.i = 0

    def setup(self):
        global initial_state
        playerax = self.ax[1]
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(playerax)
        sax = divider.append_axes("bottom", size="400%", pad=0.05)
        fax = divider.append_axes("bottom", size="400%", pad=0.05)
        sliderax_x = divider.append_axes("bottom", size="500%", pad=0.05)
        sliderax_y = divider.append_axes("bottom", size="500%", pad=0.05)
        sliderax_z = divider.append_axes("bottom", size="500%", pad=0.05)
        sliderax_vx = divider.append_axes("bottom", size="500%", pad=0.05)
        sliderax_vy = divider.append_axes("bottom", size="500%", pad=0.05)
        sliderax_vz = divider.append_axes("bottom", size="500%", pad=0.05)
        sliderax_sx = divider.append_axes("bottom", size="500%", pad=0.05)
        sliderax_sy = divider.append_axes("bottom", size="500%", pad=0.05)
        sliderax_sz = divider.append_axes("bottom", size="500%", pad=0.05)
        uax = divider.append_axes("bottom", size="400%", pad=0.05)
        playerax.remove()
        self.button_stop = matplotlib.widgets.Button(sax, label="$\u25A0$")
        self.button_forward = matplotlib.widgets.Button(fax, label="$\u25B6$")
        self.slider_x = matplotlib.widgets.Slider(
            sliderax_x,
            f"x ({-margin} - {0}m)",
            -margin,
            0,
            valinit=initial_state[0],
            color="b",
        )
        self.slider_x.label.set_color("w")
        self.slider_x.valtext.set_color("w")
        self.slider_y = matplotlib.widgets.Slider(
            sliderax_y,
            f"y ({0} - {ytable}m)",
            0,
            ytable,
            valinit=initial_state[1],
            color="b",
        )
        self.slider_y.label.set_color("w")
        self.slider_y.valtext.set_color("w")
        self.slider_z = matplotlib.widgets.Slider(
            sliderax_z,
            f"z ({0} - {1}m)",
            0,
            1.0,
            valinit=initial_state[2],
            color="b",
        )
        self.slider_z.label.set_color("w")
        self.slider_z.valtext.set_color("w")
        self.slider_vx = matplotlib.widgets.Slider(
            sliderax_vx,
            "vx (-10 - 10 m/s)",
            -10,
            10,
            valinit=initial_state[3],
            color="g",
        )
        self.slider_vx.label.set_color("w")
        self.slider_vx.valtext.set_color("w")
        self.slider_vy = matplotlib.widgets.Slider(
            sliderax_vy,
            "vy (-10 - 10 m/s)",
            -10,
            10,
            valinit=initial_state[4],
            color="g",
        )
        self.slider_vy.label.set_color("w")
        self.slider_vy.valtext.set_color("w")
        self.slider_vz = matplotlib.widgets.Slider(
            sliderax_vz,
            "vz (-10 - 10 m/s)",
            -10,
            10,
            valinit=initial_state[5],
            color="g",
        )
        self.slider_vz.label.set_color("w")
        self.slider_vz.valtext.set_color("w")
        self.slider_sx = matplotlib.widgets.Slider(
            sliderax_sx,
            f"turnspin (-{rpsmax} - {rpsmax} rps)",
            -rpsmax,
            rpsmax,
            valinit=initial_state[6] / (2 * np.pi),
            color="r",
        )
        self.slider_sx.label.set_color("w")
        self.slider_sx.valtext.set_color("w")
        self.slider_sy = matplotlib.widgets.Slider(
            sliderax_sy,
            f"topspin (-{rpsmax} - {rpsmax} rps)",
            -rpsmax,
            rpsmax,
            valinit=-initial_state[7] / (2 * np.pi),
            color="r",
        )
        self.slider_sy.label.set_color("w")
        self.slider_sy.valtext.set_color("w")
        self.slider_sz = matplotlib.widgets.Slider(
            sliderax_sz,
            f"sidespin (-{rpsmax} - {rpsmax} rps)",
            -rpsmax,
            rpsmax,
            valinit=initial_state[8] / (2 * np.pi),
            color="r",
        )
        self.slider_sz.label.set_color("w")
        self.slider_sz.valtext.set_color("w")
        self.button_update = matplotlib.widgets.Button(uax, label="$\u21BA$")
        self.button_stop.on_clicked(self.stop)
        self.button_forward.on_clicked(self.forward)
        self.slider_x.on_changed(lambda val: self.update_state(val, 0))
        self.slider_y.on_changed(lambda val: self.update_state(val, 1))
        self.slider_z.on_changed(lambda val: self.update_state(val, 2))
        self.slider_vx.on_changed(lambda val: self.update_state(val, 3))
        self.slider_vy.on_changed(lambda val: self.update_state(val, 4))
        self.slider_vz.on_changed(lambda val: self.update_state(val, 5))
        self.slider_sx.on_changed(lambda val: self.update_state(val, 6))
        self.slider_sy.on_changed(lambda val: self.update_state(-val, 7))
        self.slider_sz.on_changed(lambda val: self.update_state(val, 8))
        self.button_update.on_clicked(self.regenerate)


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("-g", "--gif", action="store_true")
    argparser.add_argument("-o", "--output_path", type=str, default=None)
    argparser.add_argument("-ns", "--no_slip", action="store_true")
    args = argparser.parse_args()
    if args.no_slip:
        apply_slip = False

    # Set up the plots and elements
    fig = plt.figure(figsize=(16, 9), constrained_layout=True, num="tt_simulator")
    gs = fig.add_gridspec(nrows=1, ncols=6)
    fig.set_facecolor("k")
    ax0 = fig.add_subplot(gs[:4], projection="3d", computed_zorder=False)
    ax1 = fig.add_subplot(gs[4:])
    ax = [ax0, ax1]
    ax[0].set_facecolor("k")
    ax[0].xaxis.pane.fill = False
    ax[0].yaxis.pane.fill = False
    ax[0].zaxis.pane.fill = False
    ax[0].xaxis.pane.set_edgecolor("k")
    ax[0].yaxis.pane.set_edgecolor("k")
    ax[0].zaxis.pane.set_edgecolor("k")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_zlabel("z")
    ax[0].xaxis.label.set_color("w")
    ax[0].yaxis.label.set_color("w")
    ax[0].zaxis.label.set_color("w")
    plot_table(ax[0])
    (ball,) = ax[0].plot(
        [initial_state[0]],
        [initial_state[1]],
        [initial_state[2]],
        c="w",
        marker=".",
        markersize=15,
    )
    (ball_preview,) = ax[0].plot(
        [initial_state[0]],
        [initial_state[1]],
        [initial_state[2]],
        c="g",
        marker=".",
        markersize=15,
    )
    (v_preview,) = ax[0].plot(
        [initial_state[0], initial_state[0] + initial_state[3] / 10],
        [initial_state[1], initial_state[1] + initial_state[4] / 10],
        [initial_state[2], initial_state[2] + initial_state[5] / 10],
        "g--",
    )
    (ln,) = ax[0].plot(
        positions[0],
        positions[1],
        positions[2],
        "w--",
        linewidth=1,
        alpha=0.5,
    )
    (trail,) = ax[0].plot([], [], [], "r", linewidth=2)
    ax[0].set_xlim(xrange)
    ax[0].set_aspect("equal")
    ax[0].view_init(ax[0].elev, 45)

    # Update function for the animation
    def update(i):
        ball.set_data_3d(
            [positions[0][i]],
            [positions[1][i]],
            [positions[2][i]],
        )
        ball_preview.set_data_3d(
            [initial_state[0]],
            [initial_state[1]],
            [initial_state[2]],
        )
        v_preview.set_data_3d(
            [initial_state[0], initial_state[0] + initial_state[3] / 10],
            [initial_state[1], initial_state[1] + initial_state[4] / 10],
            [initial_state[2], initial_state[2] + initial_state[5] / 10],
        )
        trail.set_data_3d(
            positions[0][max(0, i - trail_length) : i + 1],
            positions[1][max(0, i - trail_length) : i + 1],
            positions[2][max(0, i - trail_length) : i + 1],
        )
        ln.set_data_3d(
            positions[0],
            positions[1],
            positions[2],
        )
        return [ball, ball_preview, v_preview, trail, ln]

    # Run the program
    if args.gif:
        assert args.output_path is not None
        ani = Player(fig, ax, update, frames=len(positions[0]))
        ani.save(args.output_path, fps=int(1 / dt / interval))
    else:
        ani = Player(
            fig,
            ax,
            update,
        )
        plt.show()
