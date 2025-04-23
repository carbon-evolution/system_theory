import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.animation import FuncAnimation
from matplotlib import colors
import matplotlib.gridspec as gridspec

class EcosystemSimulation:
    def __init__(self):
        # Animation control
        self.animation = None
        self.is_animating = False
        self.frames = []
        
        # Initial populations
        self.init_prey = 40
        self.init_predator = 9
        self.init_apex_predator = 3
        
        # Basic parameters
        self.alpha = 0.1     # Prey birth rate
        self.beta = 0.02     # Predation rate (predator on prey)
        self.delta = 0.01    # Predator reproduction rate
        self.gamma = 0.1     # Predator death rate
        self.omega = 0.008   # Apex predation rate (apex on predator)
        self.epsilon = 0.005 # Apex predator reproduction rate
        self.zeta = 0.08     # Apex predator death rate
        
        # Environmental parameters
        self.carrying_capacity = 200  # Max prey population
        self.seasonal_strength = 0.2  # Strength of seasonal effects
        self.season_phase = 0         # Current season (0=spring, π=fall)
        self.time_steps = 400
        
        # Spatial parameters
        self.grid_size = 50
        self.diffusion_rate = 0.1
        self.spatial_mode = False
        
        # Initialize populations
        self.reset_simulation()
        
    def reset_simulation(self):
        # Reset animation state
        if self.animation is not None:
            self.animation.event_source.stop()
            self.animation = None
        self.is_animating = False
        self.frames = []
        
        if self.spatial_mode:
            # Initialize spatial grid for each species
            self.prey_grid = np.zeros((self.grid_size, self.grid_size))
            self.predator_grid = np.zeros((self.grid_size, self.grid_size))
            self.apex_grid = np.zeros((self.grid_size, self.grid_size))
            
            # Set initial populations in the center of the grid
            center = self.grid_size // 2
            span = self.grid_size // 10
            
            self.prey_grid[center-span:center+span, center-span:center+span] = self.init_prey / (4 * span**2)
            self.predator_grid[center-span+2:center+span-2, center-span+2:center+span-2] = self.init_predator / (4 * (span-2)**2)
            self.apex_grid[center-1:center+1, center-1:center+1] = self.init_apex_predator / 4
            
            self.prey_history = [np.sum(self.prey_grid)]
            self.predator_history = [np.sum(self.predator_grid)]
            self.apex_history = [np.sum(self.apex_grid)]
        else:
            # Non-spatial mode
            self.prey = self.init_prey
            self.predator = self.init_predator
            self.apex_predator = self.init_apex_predator
            
            self.prey_history = [self.prey]
            self.predator_history = [self.predator]
            self.apex_history = [self.apex_predator]
        
        self.time = np.arange(self.time_steps)
        
    def seasonal_effect(self, t):
        # Sinusoidal seasonal effect
        return self.seasonal_strength * np.sin(2 * np.pi * t / 100 + self.season_phase)
    
    def run_simulation(self):
        if self.spatial_mode:
            return self.run_spatial_simulation()
        else:
            return self.run_standard_simulation()
    
    def run_standard_simulation(self):
        prey = self.init_prey
        predator = self.init_predator
        apex = self.init_apex_predator
        
        prey_pop = [prey]
        predator_pop = [predator]
        apex_pop = [apex]
        
        for t in range(1, self.time_steps):
            season = self.seasonal_effect(t)
            
            # Prey dynamics with carrying capacity and seasonal effects
            prey_growth = (self.alpha * (1 + season)) * prey * (1 - prey / self.carrying_capacity) - self.beta * prey * predator
            
            # Predator dynamics with apex predation
            predator_growth = self.delta * prey * predator - self.gamma * predator - self.omega * predator * apex
            
            # Apex predator dynamics
            apex_growth = self.epsilon * predator * apex - self.zeta * apex
            
            prey += prey_growth
            predator += predator_growth
            apex += apex_growth
            
            # Ensure no negative populations
            prey = max(prey, 0)
            predator = max(predator, 0)
            apex = max(apex, 0)
            
            prey_pop.append(prey)
            predator_pop.append(predator)
            apex_pop.append(apex)
        
        return prey_pop, predator_pop, apex_pop
    
    def run_spatial_simulation(self):
        # Copy initial grids
        prey_grid = self.prey_grid.copy()
        predator_grid = self.predator_grid.copy()
        apex_grid = self.apex_grid.copy()
        
        prey_history = [np.sum(prey_grid)]
        predator_history = [np.sum(predator_grid)]
        apex_history = [np.sum(apex_grid)]
        
        # Clear previous frames
        self.frames = []
        
        # Discrete Laplacian kernel for diffusion
        laplacian = np.array([[0.05, 0.2, 0.05], 
                             [0.2, -1, 0.2], 
                             [0.05, 0.2, 0.05]])
        
        for t in range(1, self.time_steps):
            season = self.seasonal_effect(t)
            
            # Create new grids for the next step
            new_prey = prey_grid.copy()
            new_predator = predator_grid.copy()
            new_apex = apex_grid.copy()
            
            # Compute diffusion using convolution
            prey_diffusion = self.diffusion_rate * self.convolve(prey_grid, laplacian)
            predator_diffusion = self.diffusion_rate * self.convolve(predator_grid, laplacian)
            apex_diffusion = self.diffusion_rate * self.convolve(apex_grid, laplacian)
            
            # Apply population dynamics to each cell
            for i in range(1, self.grid_size-1):
                for j in range(1, self.grid_size-1):
                    # Local populations
                    prey_local = prey_grid[i, j]
                    predator_local = predator_grid[i, j]
                    apex_local = apex_grid[i, j]
                    
                    # Prey dynamics with carrying capacity and seasonal effects
                    prey_growth = (self.alpha * (1 + season)) * prey_local * (1 - prey_local / self.carrying_capacity) - self.beta * prey_local * predator_local
                    
                    # Predator dynamics with apex predation
                    predator_growth = self.delta * prey_local * predator_local - self.gamma * predator_local - self.omega * predator_local * apex_local
                    
                    # Apex predator dynamics
                    apex_growth = self.epsilon * predator_local * apex_local - self.zeta * apex_local
                    
                    # Apply growth and diffusion
                    new_prey[i, j] = max(0, prey_local + prey_growth + prey_diffusion[i, j])
                    new_predator[i, j] = max(0, predator_local + predator_growth + predator_diffusion[i, j])
                    new_apex[i, j] = max(0, apex_local + apex_growth + apex_diffusion[i, j])
            
            # Update grids
            prey_grid = new_prey
            predator_grid = new_predator
            apex_grid = new_apex
            
            # Record total populations
            prey_history.append(np.sum(prey_grid))
            predator_history.append(np.sum(predator_grid))
            apex_history.append(np.sum(apex_grid))
            
            # Store frame for animation
            if t % 5 == 0:  # Store every 5th frame to reduce memory usage
                combined_grid = np.stack([
                    predator_grid / (np.max(predator_grid) + 1e-10),
                    prey_grid / (np.max(prey_grid) + 1e-10),
                    apex_grid / (np.max(apex_grid) + 1e-10)
                ], axis=-1)
                self.frames.append(combined_grid)
        
        self.prey_grid = prey_grid
        self.predator_grid = predator_grid
        self.apex_grid = apex_grid
        
        return prey_history, predator_history, apex_history
    
    def convolve(self, grid, kernel):
        # Simple convolution implementation
        result = np.zeros_like(grid)
        k_size = kernel.shape[0]
        pad = k_size // 2
        
        for i in range(pad, grid.shape[0] - pad):
            for j in range(pad, grid.shape[1] - pad):
                window = grid[i-pad:i+pad+1, j-pad:j+pad+1]
                result[i, j] = np.sum(window * kernel)
        
        return result

# Create the simulation
sim = EcosystemSimulation()

# Create figure and layout with adjusted size and spacing
fig = plt.figure(figsize=(16, 10))
plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.2)

# Create main grid with proper spacing
gs = gridspec.GridSpec(2, 3, width_ratios=[2, 1, 1], height_ratios=[3, 1])
gs.update(wspace=0.3, hspace=0.3)

# Main population plot
ax_pop = plt.subplot(gs[0, 0])
ax_pop.set_xlabel("Time")
ax_pop.set_ylabel("Population")
ax_pop.set_title("Dynamic Ecosystem Simulation")
ax_pop.grid(True)

# Phase plot (predator vs prey)
ax_phase = plt.subplot(gs[0, 1])
ax_phase.set_xlabel("Prey Population")
ax_phase.set_ylabel("Predator Population")
ax_phase.set_title("Phase Space Plot")
ax_phase.grid(True)

# Spatial view (only shown in spatial mode)
ax_spatial = plt.subplot(gs[0, 2])
ax_spatial.set_title("Spatial Distribution")
ax_spatial.set_xticks([])
ax_spatial.set_yticks([])

# Controls area
ax_controls = plt.subplot(gs[1, :])
ax_controls.set_visible(False)

# Run initial simulation
prey_pop, predator_pop, apex_pop = sim.run_simulation()

# Set up the plots
line_prey, = ax_pop.plot(prey_pop, 'g-', label="Prey (Rabbits)")
line_predator, = ax_pop.plot(predator_pop, 'r-', label="Predators (Foxes)")
line_apex, = ax_pop.plot(apex_pop, 'b-', label="Apex Predators (Wolves)")
ax_pop.legend()

# Phase plot
phase_plot = ax_phase.scatter(prey_pop[:-10], predator_pop[:-10], c=range(len(prey_pop)-10), cmap='viridis', s=5, alpha=0.5)
phase_end = ax_phase.scatter(prey_pop[-10:], predator_pop[-10:], c=range(len(prey_pop)-10, len(prey_pop)), cmap='viridis', s=15)

# Initialize spatial view
spatial_img = None
if sim.spatial_mode:
    combined_grid = np.stack([
        sim.predator_grid / (np.max(sim.predator_grid) + 1e-10),
        sim.prey_grid / (np.max(sim.prey_grid) + 1e-10),
        sim.apex_grid / (np.max(sim.apex_grid) + 1e-10)
    ], axis=-1)
    spatial_img = ax_spatial.imshow(combined_grid, interpolation='nearest')
else:
    # Create an empty image for later use
    empty_grid = np.zeros((sim.grid_size, sim.grid_size, 3))
    spatial_img = ax_spatial.imshow(empty_grid, interpolation='nearest')
ax_spatial.set_visible(sim.spatial_mode)

# Add sliders with adjusted positions
slider_specs = [
    ("alpha", "Prey Birth Rate", 0.01, 0.5, sim.alpha, 0.15),
    ("beta", "Predation Rate", 0.001, 0.1, sim.beta, 0.13),
    ("delta", "Predator Growth", 0.001, 0.1, sim.delta, 0.11),
    ("gamma", "Predator Death", 0.01, 0.5, sim.gamma, 0.09),
    ("omega", "Apex Predation", 0.001, 0.1, sim.omega, 0.07),
    ("epsilon", "Apex Growth", 0.001, 0.1, sim.epsilon, 0.05),
    ("zeta", "Apex Death", 0.01, 0.5, sim.zeta, 0.03),
]

# Environmental controls in a separate column
env_slider_specs = [
    ("carrying_capacity", "Carrying Capacity", 50, 500, sim.carrying_capacity, 0.15),
    ("seasonal_strength", "Seasonal Effect", 0, 0.5, sim.seasonal_strength, 0.13),
    ("diffusion_rate", "Diffusion Rate", 0, 0.5, sim.diffusion_rate, 0.11),
]

sliders = {}
# Population parameter sliders (left column)
for name, label, min_val, max_val, init_val, pos in slider_specs:
    ax = plt.axes([0.1, pos, 0.25, 0.02])
    sliders[name] = Slider(ax, label, min_val, max_val, valinit=init_val)

# Environmental parameter sliders (right column)
for name, label, min_val, max_val, init_val, pos in env_slider_specs:
    ax = plt.axes([0.55, pos, 0.25, 0.02])
    sliders[name] = Slider(ax, label, min_val, max_val, valinit=init_val)

# Add control buttons with adjusted positions
btn_toggle = Button(plt.axes([0.1, 0.19, 0.15, 0.03]), 'Spatial Mode: OFF')
btn_reset = Button(plt.axes([0.3, 0.19, 0.15, 0.03]), 'Reset')
btn_play = Button(plt.axes([0.55, 0.19, 0.15, 0.03]), 'Play')
btn_stop = Button(plt.axes([0.75, 0.19, 0.15, 0.03]), 'Stop')

# Update function for the simulation
def update_simulation(_=None):
    # Update simulation parameters
    for name, slider in sliders.items():
        setattr(sim, name, slider.val)
    
    # Run simulation with new parameters
    prey_pop, predator_pop, apex_pop = sim.run_simulation()
    
    # Update plots
    line_prey.set_ydata(prey_pop)
    line_predator.set_ydata(predator_pop)
    line_apex.set_ydata(apex_pop)
    
    # Update phase plot
    phase_plot.set_offsets(np.column_stack([prey_pop[:-10], predator_pop[:-10]]))
    phase_end.set_offsets(np.column_stack([prey_pop[-10:], predator_pop[-10:]]))
    
    # Update spatial view if in spatial mode
    if sim.spatial_mode:
        combined_grid = np.stack([
            sim.predator_grid / (np.max(sim.predator_grid) + 1e-10),
            sim.prey_grid / (np.max(sim.prey_grid) + 1e-10),
            sim.apex_grid / (np.max(sim.apex_grid) + 1e-10)
        ], axis=-1)
        spatial_img.set_array(combined_grid)
        ax_spatial.set_visible(True)
    else:
        ax_spatial.set_visible(False)
    
    # Adjust y axis limits
    ax_pop.set_ylim(0, max(max(prey_pop), max(predator_pop), max(apex_pop)) * 1.1)
    ax_phase.set_xlim(0, max(prey_pop) * 1.1)
    ax_phase.set_ylim(0, max(predator_pop) * 1.1)
    
    fig.canvas.draw_idle()

# Reset function
def reset(_):
    # Reset all sliders to initial values
    for name, label, min_val, max_val, init_val, pos in slider_specs:
        sliders[name].reset()
    
    # Reset simulation
    sim.reset_simulation()
    update_simulation()

# Toggle spatial mode
def toggle_spatial_mode(_):
    global spatial_img
    
    # Toggle the mode
    sim.spatial_mode = not sim.spatial_mode
    btn_toggle.label.set_text(f'Spatial Mode: {"ON" if sim.spatial_mode else "OFF"}')
    
    # Reset simulation with new mode
    sim.reset_simulation()
    
    # Update spatial view
    if sim.spatial_mode:
        combined_grid = np.stack([
            sim.predator_grid / (np.max(sim.predator_grid) + 1e-10),
            sim.prey_grid / (np.max(sim.prey_grid) + 1e-10),
            sim.apex_grid / (np.max(sim.apex_grid) + 1e-10)
        ], axis=-1)
        if spatial_img is None:
            spatial_img = ax_spatial.imshow(combined_grid, interpolation='nearest')
        else:
            spatial_img.set_array(combined_grid)
    else:
        if spatial_img is not None:
            spatial_img.set_array(np.zeros((sim.grid_size, sim.grid_size, 3)))
    
    # Update visibility
    ax_spatial.set_visible(sim.spatial_mode)
    
    # Run simulation and update plots
    update_simulation()
    
    # Redraw
    plt.draw()

# Animation control functions
def play_animation(_):
    global spatial_img
    
    if not sim.spatial_mode:
        toggle_spatial_mode(None)  # Switch to spatial mode if not already
        return
    
    if not hasattr(sim, 'frames') or len(sim.frames) == 0:
        # Run simulation to generate frames if needed
        sim.run_simulation()
        if len(sim.frames) == 0:
            print("No frames generated for animation")
            return
    
    if sim.is_animating:
        return
    
    sim.is_animating = True
    btn_play.label.set_text('Playing...')
    
    def update_frame(frame_num):
        if not sim.is_animating:
            return [spatial_img]
        try:
            frame = sim.frames[frame_num % len(sim.frames)]
            spatial_img.set_array(frame)
            return [spatial_img]
        except Exception as e:
            print(f"Animation error: {e}")
            return [spatial_img]
    
    sim.animation = FuncAnimation(
        fig, 
        update_frame,
        frames=len(sim.frames),
        interval=50,  # 50ms between frames
        blit=True,
        repeat=True
    )
    
    plt.draw()

def stop_animation(_):
    if sim.animation is not None:
        sim.animation.event_source.stop()
        sim.animation = None
    sim.is_animating = False
    btn_play.label.set_text('Play')
    plt.draw()

# Connect callbacks
for slider in sliders.values():
    slider.on_changed(update_simulation)

btn_reset.on_clicked(reset)
btn_toggle.on_clicked(toggle_spatial_mode)
btn_play.on_clicked(play_animation)
btn_stop.on_clicked(stop_animation)

# Set initial view
plt.tight_layout()
update_simulation()

# Add window close handler
def on_close(event):
    if sim.animation is not None:
        sim.animation.event_source.stop()
        sim.animation = None

fig.canvas.mpl_connect('close_event', on_close)

plt.show()