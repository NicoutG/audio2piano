import mido
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.animation as animation
import time
import pygame
import tempfile
import tkinter as tk
from tkinter.filedialog import askopenfilename
from matplotlib.patches import Rectangle
import scipy.io.wavfile
import os
import matplotlib.cm as cm

# =========================
# PARAMETERS
# =========================
NOTE_MIN = 21
NOTE_MAX = 108
N_NOTES = NOTE_MAX - NOTE_MIN + 1

FPS = 60
WINDOW_SECONDS = 5.0
SCROLL_SPEED = 0.5

PLAY_SOUND = True
SR = 22050
window_frames = int(WINDOW_SECONDS * FPS)

WHITE_KEY_PIXEL_WIDTH = 5
BLACK_KEY_PIXEL_WIDTH = 3

GRID_COLOR = (40, 40, 40)
GRID_THICKNESS = 1

# =========================
# GLOBALS
# =========================
roll = None
full_img = None

T = window_frames
TOTAL_TIME = WINDOW_SECONDS

scroll_pos = 0.0
play_flag = False
play_start_time = None
current_play_time = 0.0

mid = None
wav_file = None
img = None

white_keys = []
black_keys = []

# =========================
# AUDIO
# =========================
pygame.mixer.init(frequency=SR, size=-16, channels=1)

def sine_with_envelope(note, dur, sr):
    n_samples = int(dur*sr)
    t = np.linspace(0, dur, n_samples, endpoint=False)
    freq = 440.0 * 2 ** ((note - 69)/12)
    env = np.ones_like(t)
    attack = int(0.01*sr)
    decay = int(0.05*sr)
    if len(t) > attack+decay:
        env[:attack] = np.linspace(0,1,attack)
        env[-decay:] = np.linspace(1,0,decay)
    wave = np.sin(2*np.pi*freq*t) * env * 0.3
    return wave

def sine_with_harmonics(note, dur, sr):
    base = sine_with_envelope(note,dur,sr)
    t = np.linspace(0, dur, int(dur*sr), endpoint=False)
    freq = 440.0 * 2 ** ((note - 69)/12)
    harmonics = 0.1*np.sin(2*np.pi*2*freq*t) + 0.05*np.sin(2*np.pi*3*freq*t)
    wave = base + harmonics
    return wave

def midi_to_wav_array(mid, sr=SR):
    active_notes = {}
    events = []
    current_time = 0.0
    for msg in mid:
        current_time += msg.time
        if msg.type=='note_on' and getattr(msg,'velocity',0)>0:
            active_notes[msg.note] = current_time
        elif msg.type in ('note_off',) or (msg.type=='note_on' and getattr(msg,'velocity',0)==0):
            if msg.note in active_notes:
                start = active_notes.pop(msg.note)
                events.append((start,msg.note,current_time-start))
    if not events:
        return np.array([]), sr
    total_duration = max(start+dur for start,_,dur in events)
    samples = np.zeros(int(np.ceil(total_duration*sr)))
    for start,note,dur in events:
        start_idx = int(round(start*sr))
        end_idx = int(round((start+dur)*sr))
        note_wave = sine_with_harmonics(note,dur,sr)
        if len(note_wave) > end_idx-start_idx:
            note_wave = note_wave[:end_idx-start_idx]
        elif len(note_wave) < end_idx-start_idx:
            note_wave = np.pad(note_wave,(0,end_idx-start_idx - len(note_wave)))
        samples[start_idx:end_idx] += note_wave
    max_amp = np.max(np.abs(samples))
    if max_amp>0:
        samples /= max_amp
    return samples,sr

def midi_to_temp_wav(mid):
    samples,sr = midi_to_wav_array(mid)
    tmp = tempfile.NamedTemporaryFile(delete=False,suffix=".wav")
    samples_int16 = (samples*32767).astype(np.int16)
    scipy.io.wavfile.write(tmp.name,sr,samples_int16)
    return tmp.name

root = tk.Tk()
root.withdraw()

# =========================
# FIGURE
# =========================
fig = plt.figure(figsize=(12,8))

manager = plt.get_current_fig_manager()
try:
    manager.window.wm_geometry("+{}+{}".format(
        int(manager.window.winfo_screenwidth()/2 - 600),
        int(manager.window.winfo_screenheight()/2 - 400)
    ))
except Exception:
    pass

ax_keys = plt.axes([0.05, 0.10, 0.90, 0.10])
ax_roll = plt.axes([0.05, 0.20, 0.90, 0.75])

# =========================
# KEYBOARD
# =========================
WHITE_PATTERN = [0,2,4,5,7,9,11]
BLACK_PATTERN = [1,3,6,8,10]

def build_keyboard():
    global white_keys, black_keys
    white_keys.clear()
    black_keys.clear()
    ax_keys.clear()
    ax_keys.set_xlim(0,52)
    ax_keys.set_ylim(0,1)
    ax_keys.axis("off")
    white_index = 0
    white_positions = {}
    for midi_note in range(NOTE_MIN, NOTE_MAX+1):
        note_mod = midi_note % 12
        if note_mod in WHITE_PATTERN:
            rect = Rectangle((white_index,0),1,1,facecolor="white",edgecolor="black")
            ax_keys.add_patch(rect)
            white_keys.append((midi_note,rect))
            white_positions[midi_note]=white_index
            white_index +=1
    for midi_note in range(NOTE_MIN, NOTE_MAX+1):
        note_mod = midi_note%12
        if note_mod in BLACK_PATTERN:
            left_white = midi_note-1
            if left_white in white_positions:
                x = white_positions[left_white]+0.7
                rect = Rectangle((x,0.4),0.6,0.6,facecolor="black",edgecolor="black")
                ax_keys.add_patch(rect)
                black_keys.append((midi_note,rect))

build_keyboard()

col_widths = []
for midi_note in range(NOTE_MIN, NOTE_MAX+1):
    if (midi_note % 12) in WHITE_PATTERN:
        col_widths.append(WHITE_KEY_PIXEL_WIDTH)
    else:
        col_widths.append(BLACK_KEY_PIXEL_WIDTH)
total_pixel_width = sum(col_widths)

placeholder = np.zeros((window_frames, total_pixel_width, 3), dtype=np.uint8)
img = ax_roll.imshow(placeholder, aspect="auto", origin="lower", interpolation='nearest')
ax_roll.set_title("Piano Roll MIDI")
ax_roll.set_xticks([])
ax_roll.set_xlabel("")

# =========================
# PRE-RENDERING HELPERS
# =========================
cmap = cm.get_cmap('inferno')

def overlay_note_grid(img_rgb, widths, color=GRID_COLOR, thickness=GRID_THICKNESS, alpha=0.45):
    out = img_rgb.astype(np.float32)
    h, w, _ = out.shape
    x = 0
    color_arr = np.array(color, dtype=np.float32)
    for col_w in widths:
        for t in range(thickness):
            xi = x + t
            if xi >= w:
                continue
            out[:, xi, :] = out[:, xi, :] * (1.0 - alpha) + color_arr * alpha
        x += col_w
    for t in range(thickness):
        xi = w - 1 - t
        if xi >= 0:
            out[:, xi, :] = out[:, xi, :] * (1.0 - alpha) + color_arr * alpha
    return np.clip(out, 0, 255).astype(np.uint8)

def render_full_image_from_roll(roll_matrix, widths):
    arr = np.clip(roll_matrix, 0.0, 1.0)
    rgba = cmap(arr)
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    T_local, N_local = rgb.shape[0], rgb.shape[1]
    total_w = sum(widths)
    out = np.zeros((T_local, total_w, 3), dtype=np.uint8)
    x = 0
    for i in range(N_local):
        w = widths[i]
        out[:, x:x+w, :] = np.repeat(rgb[:, i:i+1, :], w, axis=1)
        x += w
    out = overlay_note_grid(out, widths, color=GRID_COLOR, thickness=GRID_THICKNESS, alpha=0.35)
    return out

# =========================
# UPDATE DISPLAY
# =========================
def update_keyboard():
    """
    Use roll values (float 0..1) to color keys so repeated notes and stages are visible.
    """
    global roll
    if roll is None:
        return
    t_frame = int(current_play_time*FPS)
    if t_frame >= T:
        t_frame = T-1
    active = roll[t_frame]
    for midi_note,rect in white_keys:
        val = float(active[midi_note-NOTE_MIN])
        rect.set_facecolor((1.0, 1.0-0.5*val, 1.0-val))
    for midi_note,rect in black_keys:
        val = float(active[midi_note-NOTE_MIN])
        rect.set_facecolor((val, 0.0, 0.0))

def update_display():
    global full_img, scroll_pos
    if full_img is None:
        return
    sp = int(scroll_pos)
    sp = max(0, min(sp, T - window_frames))
    slice_img = full_img[sp:sp+window_frames]
    if slice_img.shape[0] < window_frames:
        pad_rows = window_frames - slice_img.shape[0]
        pad = np.zeros((pad_rows, total_pixel_width, 3), dtype=np.uint8)
        slice_img = np.vstack([slice_img, pad])
    img.set_data(slice_img)
    ys = np.linspace(0, window_frames-1, 6, dtype=int)
    ax_roll.set_yticks(ys)
    ax_roll.set_yticklabels([f"{(sp+y)/FPS:.2f}" for y in ys])
    update_keyboard()
    fig.canvas.draw_idle()

# =========================
# EVENTS
# =========================
def on_scroll(event):
    global scroll_pos
    direction = -1 if getattr(event, "button", "up") == "up" else 1
    scroll_pos += SCROLL_SPEED * FPS * direction
    scroll_pos = max(0, min(scroll_pos, max(0, T - window_frames)))
    update_display()

def on_mouse(event):
    global current_play_time, play_start_time, scroll_pos
    if event.inaxes != ax_roll or event.ydata is None:
        return
    t = (scroll_pos + int(event.ydata)) / FPS
    if event.button == 1:
        current_play_time = max(0.0, min(t, TOTAL_TIME))
        scroll_pos = current_play_time * FPS - window_frames // 2
        scroll_pos = max(0, min(scroll_pos, max(0, T - window_frames)))
        if play_flag and PLAY_SOUND and wav_file:
            play_start_time = time.time() - current_play_time
            try:
                pygame.mixer.music.play(start=current_play_time)
            except Exception:
                pygame.mixer.music.stop()
                pygame.mixer.music.play()
        update_display()

fig.canvas.mpl_connect("scroll_event", on_scroll)
fig.canvas.mpl_connect("button_press_event", on_mouse)

# =========================
# PLAY / PAUSE
# =========================
def play_pause(event):
    global play_flag, play_start_time, current_play_time
    if not play_flag:
        play_flag = True
        play_start_time = time.time() - current_play_time
        if PLAY_SOUND and wav_file:
            try:
                pygame.mixer.music.play(start=current_play_time)
            except Exception:
                pygame.mixer.music.play()
    else:
        play_flag = False
        current_play_time = time.time() - play_start_time
        if PLAY_SOUND and wav_file:
            pygame.mixer.music.pause()

# =========================
# RESET
# =========================
def reset(event=None):
    global play_flag, current_play_time, play_start_time, scroll_pos
    play_flag = False
    current_play_time = 0.0
    play_start_time = None
    scroll_pos = 0.0
    if PLAY_SOUND and wav_file:
        pygame.mixer.music.stop()
    update_display()

def load_midi(path):
    global roll, full_img, T, TOTAL_TIME, mid, wav_file, img, scroll_pos
    mid = mido.MidiFile(path)
    notes = []
    ongoing = {}
    current_time = 0.0
    for msg in mid:
        current_time += msg.time
        if msg.type == "note_on" and getattr(msg, 'velocity', 0) > 0:
            ongoing[msg.note] = current_time
        elif msg.type in ("note_off",) or (msg.type == "note_on" and getattr(msg, 'velocity', 0) == 0):
            if msg.note in ongoing:
                notes.append((ongoing.pop(msg.note), current_time, msg.note))
    TOTAL_TIME = current_time
    T = max(int(TOTAL_TIME * FPS) + 1, window_frames)
    roll = np.zeros((T, N_NOTES), dtype=np.float32)
    for start, end, pitch in notes:
        if not (NOTE_MIN <= pitch <= NOTE_MAX):
            continue
        start_frame = int(start * FPS)
        end_frame = int(end * FPS)
        end_frame = max(start_frame + 1, end_frame)
        end_frame = min(end_frame, T)
        duration = end_frame - start_frame
        idx = pitch - NOTE_MIN
        for i, t in enumerate(range(start_frame, end_frame)):
            alpha = i / duration
            roll[t, idx] = max(roll[t, idx], 1.0 - 0.5 * alpha)
    full_img = render_full_image_from_roll(roll, col_widths)
    if PLAY_SOUND:
        wav_file = midi_to_temp_wav(mid)
        try:
            pygame.mixer.music.load(wav_file)
        except Exception:
            wav_file = None
    scroll_pos = 0.0
    sp = 0
    slice_img = full_img[sp:sp+window_frames]
    if slice_img.shape[0] < window_frames:
        pad_rows = window_frames - slice_img.shape[0]
        pad = np.zeros((pad_rows, total_pixel_width, 3), dtype=np.uint8)
        slice_img = np.vstack([slice_img, pad])
    img.set_data(slice_img)
    build_keyboard()
    update_display()
    fig.canvas.draw_idle()

def choose_midi(event):
    path = askopenfilename(filetypes=[("MIDI files","*.mid *.midi")])
    if path:
        load_midi(path)

ax_play = plt.axes([0.25,0.02,0.15,0.06])
ax_reset= plt.axes([0.45,0.02,0.15,0.06])
ax_load = plt.axes([0.65,0.02,0.30,0.06])

btn_play = Button(ax_play,"Play / Pause")
btn_reset= Button(ax_reset,"Reset")
btn_load = Button(ax_load,"Select MIDI file")

btn_play.on_clicked(play_pause)
btn_reset.on_clicked(reset)
btn_load.on_clicked(choose_midi)

# =========================
# ANIMATION
# =========================
def animate(_):
    global current_play_time, scroll_pos
    if play_flag and full_img is not None:
        current_play_time = time.time() - play_start_time
        if current_play_time > TOTAL_TIME:
            current_play_time = TOTAL_TIME
        scroll_pos = int(current_play_time * FPS)
        scroll_pos = max(0, min(scroll_pos, max(0, T - window_frames)))
        update_display()
    return []

ani = animation.FuncAnimation(fig, animate, interval=1000/FPS, cache_frame_data=False)
plt.show()
