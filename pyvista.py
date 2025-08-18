# inside your renderer class:
class MyRenderer(RenderPyVistaMixin):
    def __init__(self, env_size, view_rad=1.0, death_anim=10):
        self.env_size = env_size
        self.view_rad = view_rad
        self.death_anim = death_anim

# later
rgb = my_renderer.render_pv(state, elev=35.0, deg_per_step=0.5, window_w=640, window_h=480)
# rgb is an HxWx3 uint8 image


# pip install pyvista Pillow
from typing import Tuple, Optional
import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont

class RenderPyVistaMixin:
    # --------- Tunables / palette (mirrors your MPL semantics) ----------
    BG_CLR       = "white"
    TERRAIN_CLR  = (0.92, 0.96, 1.00)  # light bluish
    EDGE_CLR     = (0.75, 0.80, 0.88)
    FIRE_CLR     = (0.95, 0.25, 0.25)
    DONE_CLR     = (0.60, 0.60, 0.60)
    ALIVE_CLR    = (0.15, 0.45, 0.85)
    TEXT_CLR     = (0, 0, 0)

    def _ensure_state(self, s):
        # Your MPL code did: state = state.env_state.state
        # Keep that behavior when present.
        if hasattr(s, "env_state") and hasattr(s.env_state, "state"):
            return s.env_state.state
        return s

    # ---------- Terrain helper: build StructuredGrid from heightmap ----------
    def _build_terrain(self, Z: np.ndarray, env_size: float) -> pv.StructuredGrid:
        # Z is HxW in world units along z.
        H, W = Z.shape
        # Map the terrain footprint to [0, env_size] in x,y
        xs = np.linspace(0.0, env_size, W)
        ys = np.linspace(0.0, env_size, H)
        XX, YY = np.meshgrid(xs, ys)
        grid = pv.StructuredGrid(XX, YY, Z)
        return grid

    # ---------- Agent glyphs helper ----------
    def _build_agent_glyphs(
        self,
        positions_world: np.ndarray,
        radii: np.ndarray,
        colors_rgb01: np.ndarray,
        env_size: float,
    ) -> pv.PolyData:
        """
        positions_world: (N, 3) in world coordinates (same units as env_size)
        radii:           (N,)
        colors_rgb01:    (N, 3) floats in [0,1]
        """
        if positions_world.size == 0:
            return pv.PolyData()  # empty

        pts = pv.PolyData(positions_world)
        # Store per-point radius & RGB as scalars; use glyph for spheres.
        pts["glyph_scale"] = radii.astype(float)
        # Pack rgb as 0..255 unsigned char array for direct coloring
        rgb = np.clip(colors_rgb01 * 255.0, 0, 255).astype(np.uint8)
        pts["RGB"] = rgb

        # Small unit sphere; scale by glyph_scale
        sphere = pv.Sphere(radius=1.0, phi_resolution=18, theta_resolution=18)
        glyphs = pts.glyph(
            geom=sphere,
            scale="glyph_scale",       # scale factor per point (absolute)
            orient=False,
            factor=1.0,                # 1.0 because glyph_scale already the radius
        )
        # Attach per-vertex RGB by taking point-data from the input points.
        # vtkGlyph3D transfers point data to the output by default in PyVista.
        glyphs["RGB"] = np.repeat(rgb, sphere.n_points, axis=0)
        return glyphs

    # ---------- Single view render ----------
    def _render_view(
        self,
        terrain: pv.StructuredGrid,
        agent_glyphs: pv.PolyData,
        window_size: Tuple[int, int],
        perspective: bool,
        camera_pos: Tuple[Tuple[float, float, float],
                          Tuple[float, float, float],
                          Tuple[float, float, float]],
        show_edges: bool = True,
        terrain_opacity: float = 1.0,
        lighting: str = "light kit",
        smooth_shading: bool = True,
        scalar_mode_rgb: bool = True,
    ) -> np.ndarray:
        pl = pv.Plotter(off_screen=True, window_size=window_size)
        pl.set_background(self.BG_CLR)

        # Terrain
        pl.add_mesh(
            terrain,
            color=self.TERRAIN_CLR,
            opacity=terrain_opacity,
            show_edges=show_edges,
            edge_color=self.EDGE_CLR,
            smooth_shading=smooth_shading,
        )

        # Agents (RGB per-vertex)
        if agent_glyphs.n_points > 0:
            if scalar_mode_rgb and "RGB" in agent_glyphs.point_data:
                pl.add_mesh(agent_glyphs, scalars="RGB", rgb=True, smooth_shading=True)
            else:
                pl.add_mesh(agent_glyphs, color=self.ALIVE_CLR, smooth_shading=True)

        # Camera
        pl.camera_position = camera_pos
        pl.enable_parallel_projection(not perspective)
        if lighting:
            pl.enable_lightkit()

        img = pl.screenshot(transparent_background=False, return_img=True)
        pl.close()
        # PyVista returns RGBA or RGB; ensure RGB uint8
        if img.shape[-1] == 4:
            img = img[..., :3]
        return img.astype(np.uint8)

    # ---------- Info panel via PIL ----------
    def _render_info_panel(
        self,
        size: Tuple[int, int],
        state,
    ) -> np.ndarray:
        W, H = size
        img = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Try for a monospace font, otherwise default
        try:
            font = ImageFont.truetype("DejaVuSansMono.ttf", 16)
        except Exception:
            font = ImageFont.load_default()

        info_lines = []
        # Agents (1-indexing to match your text)
        N = len(state.p_pos) if hasattr(state, "p_pos") else 0
        for i in range(N):
            fire = ", FIRING" if bool(state.fire[i]) else ""
            if bool(state.done[i]):
                info_lines.append(f"Agent {i+1:>2}: INACTIVE.")
            else:
                info_lines.append(
                    f"Agent {i+1:>2}: Battery {int(state.batt[i]):>2}{fire}."
                )

        text = "\n".join(info_lines) if info_lines else "No agents."
        # Title lines (can be expanded with any other diagnostics you had)
        header = "information\n"
        full_text = header + "\n" + text

        draw.multiline_text(
            (int(0.08 * W), int(0.08 * H)),
            full_text,
            fill=self.TEXT_CLR,
            font=font,
            spacing=6,
        )
        return np.array(img, dtype=np.uint8)

    # ---------- Public: PyVista reimplementation ----------
    def render_pv(
        self,
        state,
        elev: float = 35.0,
        deg_per_step: float = 0.5,
        window_w: int = 640,
        window_h: int = 480,
    ) -> np.ndarray:
        """
        PyVista re-implementation of your Matplotlib renderer.
        Returns an RGB numpy array with the same triple-panel layout:
        [ perspective | stacked (top, side) | information ].

        - Perspective view azimuth = state.step * deg_per_step (degrees)
        - Orthographic top: +Z view
        - Orthographic side: +X view
        """
        state = self._ensure_state(state)

        # ----- Terrain -----
        Z = np.asarray(state.hmap)  # HxW world units
        terrain = self._build_terrain(Z, self.env_size)

        # ----- Agents -----
        p_pos = np.asarray(state.p_pos, dtype=float) if hasattr(state, "p_pos") else np.zeros((0, 3))
        N = p_pos.shape[0]

        # Fading for "done" agents (like your death_anim)
        if N > 0 and hasattr(state, "done") and hasattr(state, "done_ago"):
            done = np.asarray(state.done, dtype=bool)
            done_ago = np.asarray(state.done_ago, dtype=float)
            death_frac = np.ones(N, dtype=float)
            death_frac[done] = 1.0 - np.clip(done_ago[done] / float(self.death_anim), 0.0, 1.0)
        else:
            done = np.zeros(N, dtype=bool)
            death_frac = np.ones(N, dtype=float)

        # Color per agent (replicates your MPL logic: DONE, FIRE, else ALIVE), with fade
        if N > 0:
            fire = np.asarray(state.fire, dtype=bool) if hasattr(state, "fire") else np.zeros(N, dtype=bool)
            colors = np.tile(self.ALIVE_CLR, (N, 1)).astype(float)
            colors[fire & ~done] = self.FIRE_CLR
            colors[done] = self.DONE_CLR
            # Apply fade
            colors = colors * death_frac[:, None] + np.array([1, 1, 1]) * (1.0 - death_frac)[:, None]
        else:
            colors = np.zeros((0, 3), dtype=float)

        # Use one radius for all agents (view sphere), same as MPL: self.view_rad
        radii = np.full((N,), float(getattr(self, "view_rad", 1.0)))

        agent_glyphs = self._build_agent_glyphs(p_pos, radii, colors, self.env_size)

        # ----- Camera presets -----
        # Terrain footprint is [0, env_size]^2; put the focal point at center, elevate by mean height.
        center = np.array([self.env_size / 2.0, self.env_size / 2.0, float(np.nanmean(Z))])
        radius = np.sqrt(2) * (self.env_size / 2.0)
        dist   = max(radius * 1.6, 1e-3)

        # Perspective camera: rotate around z by azimuth = step * deg_per_step
        az_deg = float(getattr(state, "step", 0)) * float(deg_per_step)
        # Spherical camera placement w.r.t. center
        # Start at yaw=45° (diag corner), pitch by elev
        yaw0 = np.deg2rad(45.0 + az_deg)
        pitch = np.deg2rad(90.0 - elev)
        cam_dir = np.array([
            np.cos(yaw0) * np.sin(pitch),
            np.sin(yaw0) * np.sin(pitch),
            np.cos(pitch),
        ])
        persp_pos = tuple((center + cam_dir * dist).tolist())
        persp_cam = (persp_pos, tuple(center.tolist()), (0.0, 0.0, 1.0))  # (pos, focal, up)

        # Top orthographic (+Z looking down)
        top_pos  = (center[0], center[1], center[2] + dist)
        top_cam  = (top_pos, tuple(center.tolist()), (0.0, 1.0, 0.0))

        # Side orthographic (+X looking toward center)
        side_pos = (center[0] + dist, center[1], center[2])
        side_cam = (side_pos, tuple(center.tolist()), (0.0, 0.0, 1.0))

        # ----- Render three views -----
        # Use same base window for each pane; we’ll downsample orthos to match your concatenation pattern.
        W, H = int(window_w), int(window_h)

        persp_img = self._render_view(
            terrain, agent_glyphs, (W, H),
            perspective=True, camera_pos=persp_cam, show_edges=True
        )

        top_img = self._render_view(
            terrain, agent_glyphs, (W, H//2),
            perspective=False, camera_pos=top_cam, show_edges=True
        )

        side_img = self._render_view(
            terrain, agent_glyphs, (W, H//2),
            perspective=False, camera_pos=side_cam, show_edges=True
        )

        # Stack top+side vertically
        # (Your MPL code then decimated orth=orth[::2, ::2]; we skip the decimation
        # because we already used half height screenshots for each.)
        orth_img = np.vstack([top_img, side_img])

        # ----- Info panel -----
        # Make it roughly the same height as persp/orth columns and ~40% width of one pane
        info_w = max(220, int(0.4 * W))
        info_img = self._render_info_panel((info_w, H), state)

        # ----- Concatenate  [perspective | orth | info] -----
        # Ensure equal heights
        def ensure_h(img, Htarget):
            if img.shape[0] == Htarget:
                return img
            # Simple nearest scaling with PIL
            im = Image.fromarray(img)
            return np.array(im.resize((img.shape[1], Htarget), Image.NEAREST))

        persp_img = ensure_h(persp_img, H)
        orth_img  = ensure_h(orth_img,  H)
        info_img  = ensure_h(info_img,  H)

        full = np.concatenate([persp_img, orth_img, info_img], axis=1).astype(np.uint8)
        return full
