import numpy as np
import pyvista as pv

def get_coil_points(center, radius, n_turns, z_height=0.05, num_points=200):
    n_turns = max(1, int(n_turns))
    t = np.linspace(0, n_turns * 2 * np.pi, num_points)
    if n_turns > 1:
        z = np.linspace(-z_height/2, z_height/2, num_points) + center[2]
    else:
        z = np.full_like(t, center[2])
    x = center[0] + radius * np.cos(t)
    y = center[1] + radius * np.sin(t)
    return np.column_stack((x, y, z))

def calculate_B_vectorized(points, coil_points, current):
    mu0 = 4 * np.pi * 1e-7
    points = np.asarray(points)
    dl_vecs = np.diff(coil_points, axis=0)
    mid_points = (coil_points[:-1] + coil_points[1:]) / 2
    B_total = np.zeros_like(points)
    for i in range(len(mid_points)):
        r_vec = points - mid_points[i]
        r_mag = np.linalg.norm(r_vec, axis=1)
        r_mag[r_mag < 1e-6] = 1e-6
        cross_prod = np.cross(dl_vecs[i], r_vec)
        factor = (mu0 * current) / (4 * np.pi * (r_mag ** 3))
        B_total += cross_prod * factor[:, None]
    return B_total

def calculate_force_real(i1, i2, coil1_geom, coil2_geom):
    B_at_coil2 = calculate_B_vectorized(coil2_geom, coil1_geom, i1)
    
    dl_vecs = np.diff(coil2_geom, axis=0)
    
    B_avg = (B_at_coil2[:-1] + B_at_coil2[1:]) / 2
    
    cross_prod = np.cross(dl_vecs, B_avg)
    dF = i2 * cross_prod

    return np.sum(dF, axis=0)[2] 

pl = pv.Plotter(window_size=[1100, 900])
pl.set_background('white')
pl.enable_anti_aliasing()

params = {
    'i1': 15.0, 'i2': -15.0, 'z2': 0.08,
    'turns1': 10, 'turns2': 10, 'rad': 0.05
}

grid_points = pv.ImageData(dimensions=(20, 20, 40), spacing=(0.015, 0.015, 0.012), origin=(-0.15, -0.15, -0.15))
mesh_field = grid_points.cast_to_structured_grid()

actor_coil1 = None; actor_coil2 = None; actor_arrow = None
stream_actor = None; text_actor = None
show_info_text = False  

def clear_streams():
    global stream_actor
    if stream_actor is not None:
        try: pl.remove_actor(stream_actor)
        except: pass
        stream_actor = None

def update_field_visuals():
    pts = mesh_field.points
    c1_geo = get_coil_points([0,0,0], params['rad'], params['turns1'], num_points=60)
    c2_geo = get_coil_points([0,0,params['z2']], params['rad'], params['turns2'], num_points=60)
    
    B1 = calculate_B_vectorized(pts, c1_geo, params['i1'])
    B2 = calculate_B_vectorized(pts, c2_geo, params['i2'])
    mesh_field['B'] = B1 + B2
    
    try:
        streams = mesh_field.streamlines('B', n_points=200, max_time=400.0, integration_direction='both')
    except Exception:
        streams = pv.PolyData(mesh_field.points)
    return streams

def refresh_field(state):
    global stream_actor, show_info_text
    
    show_info_text = state 
    
    clear_streams()
    if state:
        pl.add_text("Loading...", position='upper_right', font_size=10, name='loading_msg')
        pl.render()
        new_streams = update_field_visuals()
        try:
            colorbar_settings = {
                'title': '|B| Kekuatan Medan (Tesla)',
                'vertical': True, 
                'position_x': 0.85, 'position_y': 0.30,
                'height': 0.40, 'width': 0.08,
                'fmt': '%.2e', 'color': 'black',
                'title_font_size': 10, 'label_font_size': 9
            }
            stream_actor = pl.add_mesh(new_streams, opacity=0.6, cmap="plasma", 
                                       show_scalar_bar=True,
                                       scalar_bar_args=colorbar_settings,
                                       line_width=2)
        except: pass
        pl.remove_actor('loading_msg')
    
    update_scene()
    pl.render()

def update_text(new_text):
    global text_actor
    if text_actor is not None:
        try: pl.remove_actor(text_actor)
        except: pass
        text_actor = None

    text_actor = pl.add_text(new_text, position=(20, 300), font_size=10, color='black')

def update_scene(value=None):
    global actor_coil1, actor_coil2, actor_arrow, text_actor
    
    new_c1_pts = get_coil_points([0,0,0], params['rad'], params['turns1'], num_points=200)
    new_c2_pts = get_coil_points([0,0,params['z2']], params['rad'], params['turns2'], num_points=200)

    new_tube1 = pv.Spline(new_c1_pts, 1000).tube(radius=0.002)
    if actor_coil1 is not None: pl.remove_actor(actor_coil1)
    actor_coil1 = pl.add_mesh(new_tube1, color="cyan", smooth_shading=True, specular=0.5)

    new_tube2 = pv.Spline(new_c2_pts, 1000).tube(radius=0.002)
    if actor_coil2 is not None: pl.remove_actor(actor_coil2)
    actor_coil2 = pl.add_mesh(new_tube2, color="orange", smooth_shading=True, specular=0.5)

    f_val = calculate_force_real(params['i1'], params['i2'], new_c1_pts, new_c2_pts)
    
    direction = (0, 0, 1) if f_val > 0 else (0, 0, -1)
    color_arrow = "green" if f_val > 0 else "red"
    scale_factor = min(max(abs(f_val) * 0.05, 0.02), 0.15)
    
    if actor_arrow is not None: pl.remove_actor(actor_arrow)
    actor_arrow = pl.add_mesh(pv.Arrow(start=(0,0,params['z2']), direction=direction, scale=scale_factor), color=color_arrow)

    if show_info_text == True:
        status = "TOLAK MENOLAK" if f_val > 0 else "TARIK MENARIK"
        
        txt = (f"—— PARAMETER SIMULASI ——\n"
               f"Jarak Z: {params['z2']:.3f} m\n"
               f"Kawat Atas: {params['i2']} A | {params['turns2']} Lilitan\n"
               f"Kawat Bawah: {params['i1']} A | {params['turns1']} Lilitan\n\n"
               f"—— HASIL FISIS ——\n"
               f"Status: {status}\n"
               f"Gaya: {abs(f_val):.5f} N")
        
        update_text(txt) 
        
    else:
        if text_actor is not None:
            pl.remove_actor(text_actor)
            text_actor = None
            
    pl.render()

def cb_i1(v): params['i1']=float(v); clear_streams(); update_scene()
def cb_i2(v): params['i2']=float(v); clear_streams(); update_scene()
def cb_z(v):  params['z2']=float(v); clear_streams(); update_scene()
def cb_n1(v): params['turns1']=int(v); clear_streams(); update_scene()
def cb_n2(v): params['turns2']=int(v); clear_streams(); update_scene()

pl.add_slider_widget(cb_i1, [-20, 20], value=params['i1'], title="Arus Bawah", pointa=(0.03,0.05), pointb=(0.30,0.05), style='modern')
pl.add_slider_widget(cb_i2, [-20, 20], value=params['i2'], title="Arus Atas", pointa=(0.35,0.05), pointb=(0.62,0.05), style='modern')
pl.add_slider_widget(cb_z, [0.03, 0.25], value=params['z2'], title="Jarak Z", pointa=(0.67,0.05), pointb=(0.95,0.05), style='modern')
pl.add_slider_widget(cb_n1, [1, 50], value=params['turns1'], title="Lilitan Bawah", pointa=(0.03,0.15), pointb=(0.30,0.15), style='modern', fmt="%.0f")
pl.add_slider_widget(cb_n2, [1, 50], value=params['turns2'], title="Lilitan Atas", pointa=(0.35,0.15), pointb=(0.62,0.15), style='modern', fmt="%.0f")

pl.add_checkbox_button_widget(refresh_field, value=False, position=(10, 700), size=30, border_size=2, color_on='blue', color_off='grey')
pl.add_text("Tampilkan Aliran Medan Magnet", position=(50, 680), font_size=9)
pesan_tambahan = "Info Tambahan:\n- Klik untuk simulasi\n- Tunggu loading..."
pl.add_text(pesan_tambahan, position=(10, 600), font_size=8, color='black')
pl.add_text("X right | Z up", position='lower_right', font_size=8)

update_scene()
pl.camera_position = 'xz'; pl.camera.azimuth = 45; pl.camera.elevation = 20
pl.show()