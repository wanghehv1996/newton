"""
Parameter viewer UI for cloth simulation.

This module provides an interactive UI extension for ViewerGL that allows
real-time control and adjustment of cloth simulation parameters.
"""

import warp as wp
import numpy as np
from datetime import datetime


class ParameterViewer:
    """
    A UI extension for ViewerGL that provides sliders for dynamic parameter control.
    
    This class can be added to any ViewerGL instance to provide real-time control of:
    - FPS (frames per second)
    - Cloth properties:
      * cloth_density - Mass density of cloth (updates immediately by scaling particle masses)
    - Contact parameters:
      * cloth_particle_radius - Particle radius for cloth collision
      * cloth_body_contact_margin - Margin for cloth-body contact detection
      * self_contact_radius - Self-contact detection radius
      * self_contact_margin - Self-contact detection margin
      * self_contact_friction - Friction coefficient for cloth self-contact
      * soft_contact_ke - Soft contact stiffness
      * soft_contact_kd - Soft contact damping
    - Cloth elasticity parameters (tri_ke, tri_ka, tri_kd, bending_ke, bending_kd)
    
    Parameter Update Mechanism:
    ---------------------------
    1. Cloth density:
       - Identifies cloth particles from triangle indices (tri_indices)
       - Updates particle_mass and particle_inv_mass arrays proportionally
       - Changes take effect immediately in the next simulation step
       - No CUDA graph recapture needed (solver reads from memory)
    
    2. Self-contact parameters (radius, margin, friction):
       - These are scalar values passed to CUDA kernels
       - Changes require full reinitialization:
         a) Clear collision detection buffers (vertex/edge collision counters)
         b) Refit BVH tree with current particle positions
         c) Recapture CUDA graph with new parameter values
       - Reinitialization happens automatically but causes a brief delay (~0.1-0.5s)
    
    3. Cloth elasticity parameters (tri_ke, tri_ka, tri_kd, bending_ke, bending_kd):
       - These are stored in model arrays (tri_materials, edge_bending_properties)
       - Updates take effect immediately even with CUDA graphs
       - CUDA graphs only store array pointers, not array contents
       - We update array contents via .assign(), which affects the next simulation step
       - No reinitialization needed - instant effect!
    
    For parameters with scientific notation (e.g., 1e-4), drag controls are used 
    for intuitive adjustment:
    - Drag left/right to change value (1% per pixel by default)
    - Hold Ctrl while dragging for slower, more precise control (0.1% per pixel)
    - Click to directly input a value
    - Hover over control to see help tooltip
    
    Usage:
        viewer = newton.viewer.ViewerGL()
        param_viewer = ParameterViewer(example_instance)
        viewer.register_ui_callback(param_viewer.render, "free")
    """
    
    def __init__(self, example):
        """Initialize the ParameterViewer.
        
        Args:
            example: The Example instance that contains the parameters to control
        """
        self.example = example
        
        # Parameter ranges (min, max, default)
        # Default values tuned for cotton fabric properties:
        # - Medium stiffness (not too soft like silk, not too stiff like denim)
        # - Good draping characteristics
        # - Moderate friction coefficient
        self.param_ranges = {
            'fps': (10, 120, 60),
            'cloth_density': (0.001, 2000.0, 0.2),
            'cloth_particle_radius': (0.001, 0.02, 0.008),
            'cloth_body_contact_margin': (0.001, 0.05, 0.01),
            'self_contact_radius': (0.0001, 0.01, 0.001),
            'self_contact_margin': (0.0001, 0.01, 0.002),
            'self_contact_friction': (0.0, 2.0, 0.8),  # Cotton: moderate friction
            'soft_contact_ke': (100.0, 10000.0, 1000.0),
            'soft_contact_kd': (1e-5, 1e-1, 5e-3),
            'tri_ke': (1.0, 1000.0, 200.0),     # Cotton: medium stretch stiffness
            'tri_ka': (1.0, 1000.0, 200.0),     # Cotton: medium area preservation
            'tri_kd': (1e-8, 1e-4, 5e-6),       # Cotton: moderate damping
            'bending_ke': (1e-6, 1.0, 8e-4),    # Cotton: moderate bending resistance
            'bending_kd': (1e-5, 1e-1, 5e-3),   # Cotton: moderate bending damping
        }
    
    def render(self, imgui):
        """
        Render the parameter control UI.
        
        Args:
            imgui: The ImGui object passed by the ViewerGL callback system
        """
        if not hasattr(self.example, 'viewer') or not self.example.viewer.ui.is_available:
            return
        
        io = self.example.viewer.ui.io
        
        # Position the parameter controls window in the bottom-right corner
        window_width = 480
        window_height = min(800, io.display_size[1] - 20)  # 自适应屏幕高度，增加到800以显示所有参数和按钮
        window_x = io.display_size[0] - window_width - 10
        window_y = io.display_size[1] - window_height - 10
        
        imgui.set_next_window_pos(imgui.ImVec2(window_x, window_y))
        imgui.set_next_window_size(imgui.ImVec2(window_width, window_height))
        
        flags = imgui.WindowFlags_.no_resize.value
        
        if imgui.begin("Parameter Controls", flags=flags):
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.5, 1.0, 0.5, 1.0))
            imgui.text("Dynamic Simulation Control")
            imgui.pop_style_color()
            imgui.separator()
            
            # FPS Control
            if imgui.collapsing_header("Performance", imgui.TreeNodeFlags_.default_open.value):
                min_val, max_val, _ = self.param_ranges['fps']
                changed, new_fps = imgui.slider_int(
                    "FPS", 
                    self.example.fps, 
                    min_val, 
                    max_val
                )
                if changed:
                    self.example.fps = new_fps
                    self.example.frame_dt = 1.0 / float(new_fps)
                    self.example.sim_dt = self.example.frame_dt / self.example.sim_substeps
                    
                    # Update rigid solver's time constant
                    if hasattr(self.example, 'rigid_solver') and self.example.rigid_solver is not None:
                        self.example.rigid_solver.contact_stiffness_time_const = self.example.sim_dt
                    
                    # Recapture CUDA graph with new time step
                    self._recapture_physics_graph()
                    
                    imgui.same_line()
                    imgui.text(f"({self.example.frame_dt*1000:.2f} ms)")
            
            # Contact Parameters
            if imgui.collapsing_header("Contact Parameters", imgui.TreeNodeFlags_.default_open.value):
                imgui.separator()
                
                # Cloth particle radius
                min_val, max_val, _ = self.param_ranges['cloth_particle_radius']
                changed, new_val = imgui.slider_float(
                    "Cloth Particle Radius",
                    self.example.cloth_particle_radius,
                    min_val,
                    max_val,
                    "%.4f"
                )
                if changed:
                    self.example.cloth_particle_radius = new_val
                    # Note: This affects add_cloth_mesh, needs model rebuild
                
                # Cloth body contact margin
                min_val, max_val, _ = self.param_ranges['cloth_body_contact_margin']
                changed, new_val = imgui.slider_float(
                    "Cloth-Body Margin",
                    self.example.cloth_body_contact_margin,
                    min_val,
                    max_val,
                    "%.4f"
                )
                if changed:
                    self.example.cloth_body_contact_margin = new_val
                
                imgui.separator()
                imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(1.0, 0.8, 0.3, 1.0))
                imgui.text_wrapped("Self-contact changes need recapture")
                imgui.pop_style_color()
                imgui.separator()
                
                # Self-contact radius
                min_val, max_val, _ = self.param_ranges['self_contact_radius']
                changed, new_val = imgui.slider_float(
                    "Radius",
                    self.example.self_contact_radius,
                    min_val,
                    max_val,
                    "%.4f"
                )
                if changed:
                    self.example.self_contact_radius = new_val
                    # Update solver if it has the attribute (VBD solver)
                    if hasattr(self.example.cloth_solver, 'self_contact_radius'):
                        self.example.cloth_solver.self_contact_radius = new_val
                        # Need to recapture physics graph for the change to take effect
                        self._recapture_physics_graph()
                
                # Self-contact margin
                min_val, max_val, _ = self.param_ranges['self_contact_margin']
                changed, new_val = imgui.slider_float(
                    "Margin",
                    self.example.self_contact_margin,
                    min_val,
                    max_val,
                    "%.4f"
                )
                if changed:
                    self.example.self_contact_margin = new_val
                    # Update solver if it has the attribute (VBD solver)
                    if hasattr(self.example.cloth_solver, 'self_contact_margin'):
                        self.example.cloth_solver.self_contact_margin = new_val
                        # Need to recapture physics graph for the change to take effect
                        self._recapture_physics_graph()
                
                # Self-contact friction
                min_val, max_val, _ = self.param_ranges['self_contact_friction']
                changed, new_val = imgui.slider_float(
                    "Friction",
                    self.example.self_contact_friction,
                    min_val,
                    max_val,
                    "%.2f"
                )
                if changed:
                    self.example.self_contact_friction = new_val
                    # Update model soft contact friction if available
                    if hasattr(self.example.model, 'soft_contact_mu'):
                        self.example.model.soft_contact_mu = new_val
                        # Friction is also passed as scalar, need to recapture
                        self._recapture_physics_graph()
                
                imgui.separator()
                
                # Soft contact stiffness
                min_val, max_val, _ = self.param_ranges['soft_contact_ke']
                changed, new_val = imgui.slider_float(
                    "Soft Contact Ke",
                    self.example.soft_contact_ke,
                    min_val,
                    max_val,
                    "%.1f",
                    imgui.SliderFlags_.logarithmic.value
                )
                if changed:
                    self.example.soft_contact_ke = new_val
                    if hasattr(self.example.model, 'soft_contact_ke'):
                        self.example.model.soft_contact_ke = new_val
                
                # Soft contact damping - drag for better control
                min_val, max_val, _ = self.param_ranges['soft_contact_kd']
                changed, new_val = imgui.drag_float(
                    "Soft Contact Kd",
                    self.example.soft_contact_kd,
                    v_speed=self.example.soft_contact_kd * 0.01,  # 1% per pixel drag
                    v_min=min_val,
                    v_max=max_val,
                    format="%.2e",
                    flags=imgui.SliderFlags_.always_clamp.value
                )
                if changed:
                    self.example.soft_contact_kd = new_val
                    if hasattr(self.example.model, 'soft_contact_kd'):
                        self.example.model.soft_contact_kd = new_val
                
                # Hint for drag control
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Drag left/right to adjust. Hold Ctrl for slower. Click to input.")
            
            # Cloth Properties
            if imgui.collapsing_header("Cloth Properties", imgui.TreeNodeFlags_.default_open.value):
                imgui.separator()
                
                # Cloth density
                min_val, max_val, _ = self.param_ranges['cloth_density']
                changed, new_val = imgui.slider_float(
                    "Cloth Density",
                    self.example.CLOTH_DENSITY,
                    min_val,
                    max_val,
                    "%.3f",
                    imgui.SliderFlags_.logarithmic.value
                )
                if changed:
                    print("\n" + "="*60)
                    print("🔍 DENSITY SLIDER CHANGED!")
                    print("="*60)
                    old_density = self.example.CLOTH_DENSITY
                    self.example.CLOTH_DENSITY = new_val
                    print(f"Old density: {old_density:.3f} kg/m²")
                    print(f"New density: {new_val:.3f} kg/m²")
                    print("Calling _update_cloth_density...")
                    # Update cloth particle masses dynamically
                    self._update_cloth_density(old_density, new_val)
                    print("="*60 + "\n")
            
            # Cloth Elasticity Parameters
            if imgui.collapsing_header("Cloth Elasticity", imgui.TreeNodeFlags_.default_open.value):
                imgui.separator()
                imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.5, 1.0, 0.5, 1.0))
                imgui.text_wrapped("✓ Updates take effect immediately")
                imgui.pop_style_color()
                imgui.separator()
                
                # Triangle stiffness (elastic)
                min_val, max_val, _ = self.param_ranges['tri_ke']
                changed, new_val = imgui.slider_float(
                    "Triangle Ke (Elastic)",
                    self.example.tri_ke,
                    min_val,
                    max_val,
                    "%.1f",
                    imgui.SliderFlags_.logarithmic.value
                )
                if changed:
                    self.example.tri_ke = new_val
                    self._update_tri_materials()
                
                # Triangle stiffness (area)
                min_val, max_val, _ = self.param_ranges['tri_ka']
                changed, new_val = imgui.slider_float(
                    "Triangle Ka (Area)",
                    self.example.tri_ka,
                    min_val,
                    max_val,
                    "%.1f",
                    imgui.SliderFlags_.logarithmic.value
                )
                if changed:
                    self.example.tri_ka = new_val
                    self._update_tri_materials()
                
                # Triangle damping - drag for better control
                min_val, max_val, _ = self.param_ranges['tri_kd']
                changed, new_val = imgui.drag_float(
                    "Triangle Kd (Damping)",
                    self.example.tri_kd,
                    v_speed=self.example.tri_kd * 0.01,  # 1% per pixel drag
                    v_min=min_val,
                    v_max=max_val,
                    format="%.2e",
                    flags=imgui.SliderFlags_.always_clamp.value
                )
                if changed:
                    self.example.tri_kd = new_val
                    self._update_tri_materials()
                
                # Hint for drag control
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Drag left/right to adjust. Hold Ctrl for slower. Click to input.")
                
                imgui.separator()
                
                # Bending stiffness - drag for better control
                min_val, max_val, _ = self.param_ranges['bending_ke']
                changed, new_val = imgui.drag_float(
                    "Bending Ke (Elastic)",
                    self.example.bending_ke,
                    v_speed=self.example.bending_ke * 0.01,  # 1% per pixel drag
                    v_min=min_val,
                    v_max=max_val,
                    format="%.2e",
                    flags=imgui.SliderFlags_.always_clamp.value
                )
                if changed:
                    self.example.bending_ke = new_val
                    self._update_edge_bending_properties()
                
                # Hint for drag control
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Drag left/right to adjust. Hold Ctrl for slower. Click to input.")
                
                # Bending damping - drag for better control
                min_val, max_val, _ = self.param_ranges['bending_kd']
                changed, new_val = imgui.drag_float(
                    "Bending Kd (Damping)",
                    self.example.bending_kd,
                    v_speed=self.example.bending_kd * 0.01,  # 1% per pixel drag
                    v_min=min_val,
                    v_max=max_val,
                    format="%.2e",
                    flags=imgui.SliderFlags_.always_clamp.value
                )
                if changed:
                    self.example.bending_kd = new_val
                    self._update_edge_bending_properties()
                
                # Hint for drag control
                if imgui.is_item_hovered():
                    imgui.set_tooltip("Drag left/right to adjust. Hold Ctrl for slower. Click to input.")
            
            # Show current values summary (默认折叠以节省空间)
            imgui.separator()
            if imgui.collapsing_header("Current Values"):
                imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.7, 0.9, 1.0, 1.0))
                imgui.text(f"FPS: {self.example.fps}")
                imgui.separator()
                imgui.text("Cloth Properties:")
                imgui.pop_style_color()
                imgui.text(f"  Density: {self.example.CLOTH_DENSITY:.3f}")
                imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.7, 0.9, 1.0, 1.0))
                imgui.separator()
                imgui.text("Contact:")
                imgui.pop_style_color()
                imgui.text(f"  Particle Radius: {self.example.cloth_particle_radius:.4f}")
                imgui.text(f"  Body Margin: {self.example.cloth_body_contact_margin:.4f}")
                imgui.text(f"  Self Radius: {self.example.self_contact_radius:.4f}")
                imgui.text(f"  Self Margin: {self.example.self_contact_margin:.4f}")
                imgui.text(f"  Self Friction: {self.example.self_contact_friction:.2f}")
                imgui.text(f"  Soft Ke: {self.example.soft_contact_ke:.1f}")
                imgui.text(f"  Soft Kd: {self.example.soft_contact_kd:.2e}")
                imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.7, 0.9, 1.0, 1.0))
                imgui.separator()
                imgui.text("Cloth Elasticity:")
                imgui.pop_style_color()
                imgui.text(f"  Tri Ke: {self.example.tri_ke:.1f}")
                imgui.text(f"  Tri Ka: {self.example.tri_ka:.1f}")
                imgui.text(f"  Tri Kd: {self.example.tri_kd:.2e}")
                imgui.text(f"  Bend Ke: {self.example.bending_ke:.2e}")
                imgui.text(f"  Bend Kd: {self.example.bending_kd:.2e}")
            
            # Action buttons
            imgui.separator()
            
            # Save parameters button
            if imgui.button("Save Parameters to JSON", imgui.ImVec2(-1, 30)):
                self._save_parameters_to_json()
            
            # Reset button
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0.8, 0.3, 0.3, 1.0))
            imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(1.0, 0.4, 0.4, 1.0))
            imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0.9, 0.2, 0.2, 1.0))
            if imgui.button("Reset Simulation", imgui.ImVec2(-1, 30)):
                self._reset_simulation()
            imgui.pop_style_color(3)
        
        imgui.end()
    
    def _update_tri_materials(self):
        """Update triangle material properties in the model.
        
        Note: These updates take effect immediately even with CUDA graph capture,
        because the kernel reads from array memory, not baked-in constants.
        The CUDA graph records the pointer to tri_materials, and we're updating
        the data at that pointer location.
        """
        if not hasattr(self.example, 'model') or self.example.model.tri_materials is None:
            return
        
        # Update all triangle materials with new stiffness and damping values
        # tri_materials structure: [tri_ke, tri_ka, tri_kd, tri_drag, tri_lift]
        tri_mat_np = self.example.model.tri_materials.numpy()
        tri_mat_np[:, 0] = self.example.tri_ke  # elastic stiffness
        tri_mat_np[:, 1] = self.example.tri_ka  # area stiffness
        tri_mat_np[:, 2] = self.example.tri_kd  # damping
        # Leave tri_drag (index 3) and tri_lift (index 4) unchanged
        self.example.model.tri_materials.assign(tri_mat_np)
    
    def _update_edge_bending_properties(self):
        """Update edge bending properties in the model.
        
        Note: These updates take effect immediately even with CUDA graph capture,
        because the kernel reads from array memory, not baked-in constants.
        The CUDA graph records the pointer to edge_bending_properties, and we're
        updating the data at that pointer location.
        """
        if not hasattr(self.example, 'model') or self.example.model.edge_bending_properties is None:
            return
        
        # Update all edge bending properties with new stiffness and damping values
        # edge_bending_properties structure: [edge_ke, edge_kd]
        edge_props_np = self.example.model.edge_bending_properties.numpy()
        edge_props_np[:, 0] = self.example.bending_ke  # bending elastic stiffness
        edge_props_np[:, 1] = self.example.bending_kd  # bending damping
        self.example.model.edge_bending_properties.assign(edge_props_np)
    
    def _reset_simulation(self):
        """Reset the simulation to initial state."""
        try:
            print("\n" + "="*60)
            print("🔄 Resetting Simulation to Initial State...")
            print("="*60)
            
            # 1. Reset timing
            self.example.sim_time = 0.0
            self.example.sim_frame = 0
            self.example._substep_index = 0
            print("✓ Reset simulation time and frame counter")
            
            # 2. Reset particle states (cloth)
            # Copy initial positions from model
            self.example.state_0.particle_q.assign(self.example.model.particle_q)
            self.example.state_0.particle_qd.zero_()
            self.example.state_1.particle_q.assign(self.example.model.particle_q)
            self.example.state_1.particle_qd.zero_()
            print("✓ Reset cloth particle positions and velocities")
            
            # 3. Reset joint states (robot)
            self.example.state_0.joint_q.assign(self.example.model.joint_q)
            self.example.state_0.joint_qd.zero_()
            self.example.state_1.joint_q.assign(self.example.model.joint_q)
            self.example.state_1.joint_qd.zero_()
            
            # Evaluate forward kinematics with initial joint positions
            import newton
            newton.eval_fk(self.example.model, self.example.state_0.joint_q, 
                          self.example.state_0.joint_qd, self.example.state_0)
            print("✓ Reset robot joint positions and velocities")
            
            # 4. Reset gripper states (if available)
            if (hasattr(self.example, 'left_gripper_state') and 
                hasattr(self.example, 'open_left_gripper') and 
                hasattr(self.example, 'open_right_gripper')):
                self.example.left_gripper_state = self.example.open_left_gripper
                self.example.right_gripper_state = self.example.open_right_gripper
                print("✓ Reset gripper states")
            
            # 5. Reset end effector targets to current positions
            if (hasattr(self.example, 'gizmo_lee_tf') and hasattr(self.example, 'gizmo_ree_tf') and
                hasattr(self.example, 'left_ee_body_id') and hasattr(self.example, 'right_ee_body_id')):
                # Get current end effector transforms from state
                lee_body_id = self.example.left_ee_body_id
                ree_body_id = self.example.right_ee_body_id
                
                self.example.gizmo_lee_tf = self.example.state_0.body_q[lee_body_id]
                self.example.gizmo_ree_tf = self.example.state_0.body_q[ree_body_id]
                print("✓ Reset end effector target positions")
            
            # 6. Clear contact information
            if hasattr(self.example, 'contacts'):
                self.example.contacts = self.example.model.collide(self.example.state_0)
                print("✓ Clear contact information")
            
            # 7. Reset collision detector if available
            if hasattr(self.example, 'cloth_solver') and hasattr(self.example.cloth_solver, 'trimesh_collision_detector'):
                collision_info = self.example.cloth_solver.trimesh_collision_detector.collision_info
                if hasattr(collision_info, 'vertex_colliding_triangles_count') and collision_info.vertex_colliding_triangles_count is not None:
                    collision_info.vertex_colliding_triangles_count.zero_()
                if hasattr(collision_info, 'triangle_colliding_vertices_count') and collision_info.triangle_colliding_vertices_count is not None:
                    collision_info.triangle_colliding_vertices_count.zero_()
                if hasattr(collision_info, 'edge_colliding_edges_count') and collision_info.edge_colliding_edges_count is not None:
                    collision_info.edge_colliding_edges_count.zero_()
                print("✓ Reset collision detection buffers")
            
            print("="*60)
            print("✅ Simulation Reset Complete!")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"❌ Error resetting simulation: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_parameters_to_json(self):
        """Save all current parameters to a JSON file and print them."""
        # Generate filename with timestamp
        filename = f"./cloth_sim_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Use Example's save_physics_config method
        self.example.save_physics_config(filename)
        
        # Print current parameters to console
        print("\n" + "="*60)
        print("📊 Current Physics Parameters")
        print("="*60)
        print(f"  Contact Parameters:")
        print(f"    cloth_particle_radius: {self.example.cloth_particle_radius:.4f}")
        print(f"    cloth_body_contact_margin: {self.example.cloth_body_contact_margin:.4f}")
        print(f"    self_contact_radius: {self.example.self_contact_radius:.4f}")
        print(f"    self_contact_margin: {self.example.self_contact_margin:.4f}")
        print(f"    self_contact_friction: {self.example.self_contact_friction:.2f}")
        print(f"    soft_contact_ke: {self.example.soft_contact_ke:.1f}")
        print(f"    soft_contact_kd: {self.example.soft_contact_kd:.2e}")
        print(f"  Cloth Elasticity:")
        print(f"    tri_ke: {self.example.tri_ke:.1f}")
        print(f"    tri_ka: {self.example.tri_ka:.1f}")
        print(f"    tri_kd: {self.example.tri_kd:.2e}")
        print(f"    bending_ke: {self.example.bending_ke:.2e}")
        print(f"    bending_kd: {self.example.bending_kd:.2e}")
        print(f"  Other:")
        print(f"    robot_friction: {self.example.robot_friction:.2f}")
        print(f"    table_friction: {self.example.table_friction:.2f}")
        print("="*60 + "\n")
    
    def _update_cloth_density(self, old_density, new_density):
        """Update cloth particle masses based on new density.
        
        This method identifies cloth particles (from tri_indices) and updates their
        masses proportionally. It also updates the inverse mass array.
        
        Args:
            old_density: Previous cloth density value
            new_density: New cloth density value
        """
        print("🔍 [DEBUG] _update_cloth_density() called")
        print(f"   old_density={old_density}, new_density={new_density}")
        
        if not hasattr(self.example, 'model'):
            print("⚠ Warning: example has no 'model' attribute")
            return
        
        if self.example.model.particle_mass is None:
            print("⚠ Warning: Cannot update density - particle_mass is None")
            return
        
        print(f"🔍 [DEBUG] Model and particle_mass exist")
        
        if abs(old_density) < 1e-10:
            print("⚠ Warning: Old density too small, cannot compute ratio")
            return
        
        try:
            # Calculate density ratio
            density_ratio = new_density / old_density
            print(f"🔍 [DEBUG] Density ratio: {density_ratio:.3f}")
            
            # Get cloth particle indices from triangles
            # tri_indices contains all triangle vertex indices (3 per triangle)
            if self.example.model.tri_indices is None:
                print("⚠ Warning: No triangles found in model (tri_indices is None)")
                return
            
            print(f"🔍 [DEBUG] tri_indices exists, shape: {self.example.model.tri_indices.shape}")
            
            # Find unique particle indices used by cloth triangles
            tri_indices_np = self.example.model.tri_indices.numpy()
            cloth_particle_indices = np.unique(tri_indices_np)
            print(f"🔍 [DEBUG] Found {len(cloth_particle_indices)} unique cloth particles")
            
            # Get current particle masses
            particle_mass_np = self.example.model.particle_mass.numpy()
            print(f"🔍 [DEBUG] particle_mass array shape: {particle_mass_np.shape}")
            
            # Calculate total mass before update
            old_total_mass = np.sum([particle_mass_np[i] for i in cloth_particle_indices if 0 <= i < len(particle_mass_np)])
            print(f"🔍 [DEBUG] Total cloth mass before: {old_total_mass:.6f} kg")
            
            # Update masses for cloth particles
            updated_count = 0
            for particle_idx in cloth_particle_indices:
                if 0 <= particle_idx < len(particle_mass_np):
                    old_mass = particle_mass_np[particle_idx]
                    new_mass = old_mass * density_ratio
                    particle_mass_np[particle_idx] = new_mass
                    updated_count += 1
            
            print(f"🔍 [DEBUG] Updated {updated_count} particle masses")
            
            # Update the model's particle_mass array
            self.example.model.particle_mass.assign(particle_mass_np)
            print(f"🔍 [DEBUG] particle_mass array updated on GPU")
            
            # Update inverse mass array
            particle_inv_mass_np = self.example.model.particle_inv_mass.numpy()
            inv_mass_count = 0
            for particle_idx in cloth_particle_indices:
                if 0 <= particle_idx < len(particle_inv_mass_np):
                    if particle_mass_np[particle_idx] > 1e-10:
                        particle_inv_mass_np[particle_idx] = 1.0 / particle_mass_np[particle_idx]
                        inv_mass_count += 1
                    else:
                        particle_inv_mass_np[particle_idx] = 0.0  # kinematic particle
            
            print(f"🔍 [DEBUG] Updated {inv_mass_count} inverse masses")
            self.example.model.particle_inv_mass.assign(particle_inv_mass_np)
            print(f"🔍 [DEBUG] particle_inv_mass array updated on GPU")
            
            # Calculate and display total mass change
            total_mass = np.sum([particle_mass_np[i] for i in cloth_particle_indices])
            print(f"✓ Cloth density updated: {old_density:.3f} → {new_density:.3f} kg/m²")
            print(f"  Affected {len(cloth_particle_indices)} particles")
            print(f"  Total cloth mass: {total_mass:.4f} kg ({total_mass*1000:.1f} g)")
            
        except Exception as e:
            print(f"❌ Error updating cloth density: {e}")
            import traceback
            traceback.print_exc()
    
    def _recapture_physics_graph(self):
        """Recapture the physics CUDA graph and reinitialize collision detection.
        
        This is necessary because:
        1. CUDA graphs bake in constant values (self_contact_radius, margin, friction)
        2. Collision detector may need to reset internal buffers
        3. BVH树需要在新的参数下重新适应
        
        When solver parameters like self_contact_radius change, we need to:
        - Clear collision detection buffers
        - Rebuild BVH tree
        - Recapture the CUDA graph with new parameters
        """
        if not hasattr(self.example, 'physics_graph'):
            return
        
        # Reinitialize collision detector buffers
        if hasattr(self.example, 'cloth_solver') and hasattr(self.example.cloth_solver, 'trimesh_collision_detector'):
            # Clear collision detection counters
            collision_info = self.example.cloth_solver.trimesh_collision_detector.collision_info
            if hasattr(collision_info, 'vertex_colliding_triangles_count') and collision_info.vertex_colliding_triangles_count is not None:
                collision_info.vertex_colliding_triangles_count.zero_()
            if hasattr(collision_info, 'triangle_colliding_vertices_count') and collision_info.triangle_colliding_vertices_count is not None:
                collision_info.triangle_colliding_vertices_count.zero_()
            if hasattr(collision_info, 'edge_colliding_edges_count') and collision_info.edge_colliding_edges_count is not None:
                collision_info.edge_colliding_edges_count.zero_()
            
            # Rebuild BVH with current particle positions
            # This ensures the BVH is optimized for the current configuration
            if hasattr(self.example, 'state_0') and self.example.state_0.particle_q is not None:
                self.example.cloth_solver.trimesh_collision_detector.refit(self.example.state_0.particle_q)
        
        # Only recapture if we're using CUDA and have a graph
        if self.example.physics_graph is not None:
            if wp.get_device().is_cuda and not self.example.use_mujoco_cpu:
                try:
                    # Recapture the physics simulation graph with new parameters
                    with wp.ScopedCapture() as capture:
                        self.example.physics_simulate()
                    self.example.physics_graph = capture.graph
                    print(f"✓ Physics graph recaptured with updated self-contact parameters")
                except Exception as e:
                    print(f"⚠ Warning: Failed to recapture physics graph: {e}")
                    print(f"  Parameters updated but may require manual restart for full effect")

