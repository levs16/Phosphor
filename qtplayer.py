import sys
from PyQt6.QtWidgets import *
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
import cv2
import yt_dlp
import numpy as np
import time
from pygame import mixer
import random
import os
import hashlib
import json
from pathlib import Path

class VideoThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)
    
    def __init__(self, video_url, video_window):
        super().__init__()
        self.video_url = video_url
        self.video_window = video_window
        self.running = True
        self.playback_time = 0
        self.last_frame_time = None
        
        # Check if this is a local file or URL
        self.is_local_file = os.path.exists(video_url)
        
        # Extract video title only for URLs
        if not self.is_local_file:
            try:
                with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                    info = ydl.extract_info(video_url, download=False)
                    self.video_name = info.get('title', 'Video')
                    # Get direct video URL
                    for format in info['formats']:
                        if format.get('vcodec', 'none') != 'none':
                            self.video_url = format['url']
                            break
            except Exception as e:
                print(f"Error extracting video info: {e}")
                self.video_name = os.path.basename(video_url)
        else:
            self.video_name = os.path.basename(video_url)
        
        self.video_name = self.video_name.replace('-', ' ').replace('_', ' ')
        if len(self.video_name) > 30:
            self.video_name = self.video_name[:27] + "..."
    
    def run(self):
        try:
            cap = cv2.VideoCapture(self.video_url)
            if not cap.isOpened():
                self.error.emit("Failed to open video stream")
                return
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_delay = 1.0 / fps if fps > 0 else 1.0/30.0
            
            last_frame = None
            frame_count = 0
            
            while self.running and cap.isOpened():
                if not self.video_window.is_paused:
                    # Get current audio position
                    audio_pos = max(0, mixer.music.get_pos() / 1000.0)  # Convert to seconds
                    
                    # Calculate the target frame number based on audio position
                    target_frame = int(audio_pos * fps)
                    
                    # If we're behind or ahead by more than 1 frame, seek to the correct position
                    if abs(frame_count - target_frame) > 1:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                        frame_count = target_frame
                    
                    ret, frame = cap.read()
                    if ret:
                        frame_count += 1
                        last_frame = frame.copy()
                        frame = cv2.resize(frame, (854, 480))
                        frame = self.add_vcr_overlay(frame, audio_pos)
                        self.frame_ready.emit(frame)
                    else:
                        break
                    
                    time.sleep(frame_delay)
                else:
                    if last_frame is not None:
                        # During pause, keep displaying the last frame with updated overlay
                        paused_frame = last_frame.copy()
                        paused_frame = cv2.resize(paused_frame, (854, 480))
                        # Use the last known audio position for the overlay
                        audio_pos = max(0, mixer.music.get_pos() / 1000.0)
                        paused_frame = self.add_vcr_overlay(paused_frame, audio_pos)
                        self.frame_ready.emit(paused_frame)
                    time.sleep(0.03)  # Reduce CPU usage during pause
            
            cap.release()
            
        except Exception as e:
            self.error.emit(str(e))
            print(f"Video thread error: {str(e)}")
    
    def add_vcr_overlay(self, frame, elapsed_time):
        height, width = frame.shape[:2]
        overlay = frame.copy()
        
        # VCR green color and font setup
        VCR_GREEN = (0, 255, 0)  # BGR format
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Only show overlay during first 2 seconds or when paused
        should_show = elapsed_time <= 2.0 or self.video_window.is_paused
        
        if should_show:
            # Play/Pause symbol (top left)
            if self.video_window.is_paused:
                # Draw pause symbol
                cv2.rectangle(overlay, (30, 20), (40, 50), VCR_GREEN, -1)
                cv2.rectangle(overlay, (50, 20), (60, 50), VCR_GREEN, -1)
            else:
                # Draw play triangle
                triangle_pts = np.array([
                    [30, 20],
                    [30, 50],
                    [60, 35]
                ], np.int32)
                cv2.fillPoly(overlay, [triangle_pts], VCR_GREEN)
            
            # Video name (bottom left)
            cv2.putText(overlay, self.video_name, 
                       (30, height - 30),
                       font, 0.7,
                       VCR_GREEN,
                       2,
                       cv2.LINE_AA)
            
            # Phosphor branding (top right)
            branding = "Phosphor"
            cv2.putText(overlay, branding,
                       (width - 120, 30),
                       font, 0.7,
                       VCR_GREEN,
                       2,
                       cv2.LINE_AA)
            
            # Calculate alpha for fading
            alpha = 1.0
            if elapsed_time < 0.5:  # Fade in
                alpha = elapsed_time * 2
            elif elapsed_time > 1.5 and not self.video_window.is_paused:  # Fade out
                alpha = max(0, 2.0 - elapsed_time)
            
            # Apply overlay with alpha
            frame = cv2.addWeighted(overlay, alpha * 0.7, frame, 1 - alpha * 0.7, 0)
            
            # Add green glow
            green_glow = np.zeros_like(frame)
            if self.video_window.is_paused:
                cv2.rectangle(green_glow, (30, 20), (40, 50), (0, 50, 0), -1)
                cv2.rectangle(green_glow, (50, 20), (60, 50), (0, 50, 0), -1)
            else:
                triangle_pts = np.array([
                    [30, 20],
                    [30, 50],
                    [60, 35]
                ], np.int32)
                cv2.fillPoly(green_glow, [triangle_pts], (0, 50, 0))
            
            # Add glow to video name and branding
            cv2.putText(green_glow, self.video_name, 
                       (30, height - 30),
                       font, 0.7,
                       (0, 50, 0),
                       4,
                       cv2.LINE_AA)
            
            cv2.putText(green_glow, branding,
                       (width - 120, 30),
                       font, 0.7,
                       (0, 50, 0),
                       4,
                       cv2.LINE_AA)
                       
            frame = cv2.addWeighted(frame, 1, green_glow, alpha * 0.3, 0)
        
        return frame
    
    def stop(self):
        self.running = False

class VideoCache:
    def __init__(self):
        self.cache_dir = Path.home() / '.qtplayer_cache'
        self.cache_dir.mkdir(exist_ok=True)
        self.index_file = self.cache_dir / 'cache_index.json'
        self.load_index()

    def load_index(self):
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {}

    def save_index(self):
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f)

    def get_cache_path(self, url):
        url_hash = hashlib.md5(url.encode()).hexdigest()
        if url_hash in self.index:
            video_path = self.cache_dir / f"{url_hash}.mp4"
            audio_path = self.cache_dir / f"{url_hash}.mp3"
            if video_path.exists() and audio_path.exists():
                return str(video_path), str(audio_path)
        return None, None

    def cache_video(self, url, video_path, audio_path):
        url_hash = hashlib.md5(url.encode()).hexdigest()
        self.index[url_hash] = {
            'url': url,
            'timestamp': time.time()
        }
        self.save_index()

class VideoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.video_label = QLabel()
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.video_label)
        self.setLayout(self.layout)
        self.setMinimumSize(854, 480)
        
        # Set black background
        self.setStyleSheet("background-color: black;")
        self.video_label.setStyleSheet("background-color: black;")
        
        self.is_paused = False
        self.pause_time = None
        self.is_fullscreen = False
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space:
            self.is_paused = not self.is_paused
            if self.is_paused:
                mixer.music.pause()
                self.pause_time = mixer.music.get_pos() / 1000.0  # Store pause time in seconds
            else:
                mixer.music.unpause()
        elif event.key() == Qt.Key.Key_F:
            self.toggle_fullscreen()
        super().keyPressEvent(event)
    
    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

class ControlWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Phosphor Controls")
        self.setFixedSize(400, 500)  # Made taller to fit effects
        
        # Initialize pygame mixer
        mixer.init()
        
        # Create video window
        self.video_window = VideoWindow()
        self.video_window.show()
        
        # Main widget setup
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # URL input
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Enter YouTube URL")
        self.url_input.setStyleSheet("""
            QLineEdit {
                background-color: #000000;
                color: #00ff33;
                border: 2px solid #00ff33;
                padding: 8px;
            }
        """)
        layout.addWidget(self.url_input)
        
        # Add file button next to URL input
        url_layout = QHBoxLayout()
        url_layout.addWidget(self.url_input)
        
        self.file_button = QPushButton("LOCAL FILE")
        self.file_button.clicked.connect(self.open_local_file)
        url_layout.addWidget(self.file_button)
        
        layout.addLayout(url_layout)
        
        # Controls container
        controls_layout = QHBoxLayout()
        
        # Play button
        self.play_button = QPushButton("▶ PLAY")
        self.play_button.clicked.connect(self.start_video)
        controls_layout.addWidget(self.play_button)
        
        # Fullscreen button
        self.fullscreen_button = QPushButton("FULLSCREEN")
        self.fullscreen_button.clicked.connect(self.video_window.toggle_fullscreen)
        controls_layout.addWidget(self.fullscreen_button)
        
        layout.addLayout(controls_layout)
        
        # Quality selector
        quality_layout = QHBoxLayout()
        quality_label = QLabel("QUALITY:")
        quality_label.setStyleSheet("color: #00ff33;")
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["720p", "480p", "360p", "240p"])
        self.quality_combo.setStyleSheet("""
            QComboBox {
                background-color: #000000;
                color: #00ff33;
                border: 2px solid #00ff33;
                padding: 5px;
            }
        """)
        quality_layout.addWidget(quality_label)
        quality_layout.addWidget(self.quality_combo)
        layout.addLayout(quality_layout)
        
        # CRT Effects Group
        effects_group = QGroupBox("CRT EFFECTS")
        effects_group.setStyleSheet("""
            QGroupBox {
                color: #00ff33;
                border: 2px solid #00ff33;
                margin-top: 1em;
                padding: 10px;
            }
        """)
        effects_layout = QVBoxLayout()
        
        # Create sliders
        self.scanline_slider = self.create_slider("SCANLINES")
        self.curvature_slider = self.create_slider("CURVATURE")
        self.bloom_slider = self.create_slider("BLOOM")
        self.rgb_shift_slider = self.create_slider("RGB SHIFT")
        self.noise_slider = self.create_slider("NOISE")
        self.vignette_slider = self.create_slider("VIGNETTE")
        
        # Add sliders to effects layout
        for slider_layout in [
            self.create_slider_layout("SCANLINES", self.scanline_slider),
            self.create_slider_layout("CURVATURE", self.curvature_slider),
            self.create_slider_layout("BLOOM", self.bloom_slider),
            self.create_slider_layout("RGB SHIFT", self.rgb_shift_slider),
            self.create_slider_layout("NOISE", self.noise_slider),
            self.create_slider_layout("VIGNETTE", self.vignette_slider)
        ]:
            effects_layout.addLayout(slider_layout)
        
        effects_group.setLayout(effects_layout)
        layout.addWidget(effects_group)
        
        # Status label
        self.status_label = QLabel("READY")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                color: #00ff33;
                padding: 10px;
                background-color: #000000;
                border: 2px solid #00ff33;
            }
        """)
        layout.addWidget(self.status_label)
        
        # Set window style
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #000000;
            }
            QPushButton {
                background-color: #000000;
                color: #00ff33;
                border: 2px solid #00ff33;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #00ff33;
                color: #000000;
            }
            QLabel {
                color: #00ff33;
            }
            QSlider::groove:horizontal {
                border: 1px solid #00ff33;
                height: 8px;
                background: #000000;
            }
            QSlider::handle:horizontal {
                background: #00ff33;
                width: 18px;
                margin: -5px 0;
            }
        """)
        
        self.setup_style()
        
        # Initialize video cache
        self.video_cache = VideoCache()
        
        # Initialize video thread
        self.video_thread = None

    def setup_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #000000;
            }
            QLabel {
                color: #00ff33;
                font-family: 'Menlo';
                font-size: 14px;
            }
            QPushButton {
                background-color: #111111;
                color: #00ff33;
                border: 2px solid #00ff33;
                border-radius: 10px;
                padding: 10px;
                font-family: 'Menlo';
                font-size: 14px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #00ff33;
                color: black;
            }
            QLineEdit {
                background-color: #111111;
                color: #00ff33;
                border: 2px solid #00ff33;
                border-radius: 10px;
                padding: 5px;
                font-family: 'Menlo';
                font-size: 14px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #00ff33;
                height: 8px;
                background: #111111;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #00ff33;
                border: 1px solid #00ff33;
                width: 18px;
                height: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)

    def create_slider(self, name):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(50)
        return slider

    def create_slider_layout(self, label_text, slider):
        layout = QHBoxLayout()
        label = QLabel(label_text)
        label.setStyleSheet("color: #00ff33;")
        layout.addWidget(label)
        layout.addWidget(slider)
        return layout

    def start_video(self):
        try:
            url = self.url_input.text().strip()
            if not url:
                self.status_label.setText("ERROR: Enter URL or select file")
                return
            
            # Check if it's a local file
            if os.path.exists(url):
                self.start_cached_video(url, url)  # For local files, use same path for video and audio
                return
            
            # Handle YouTube URL
            if not url.startswith(('http://', 'https://')) and '.' in url:
                url = 'https://' + url

            self.status_label.setText("LOADING...")
            self.play_button.setEnabled(False)

            # Check cache first
            cached_video, cached_audio = self.video_cache.get_cache_path(url)
            if cached_video and cached_audio:
                self.status_label.setText("LOADING FROM CACHE...")
                self.start_cached_video(cached_video, cached_audio)
                return

            # Get selected quality
            quality = self.quality_combo.currentText()
            height = int(quality.replace('p', ''))

            # Configure yt-dlp options based on selected quality
            ydl_opts = {
                'format': f'best[height<={height}]',
                'quiet': True,
                'no_warnings': True,
            }

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    
                    # Get best format with both video and audio
                    formats = info.get('formats', [])
                    best_format = None
                    
                    for f in formats:
                        if f.get('acodec') != 'none' and f.get('vcodec') != 'none':
                            best_format = f
                            break
                    
                    if not best_format:
                        raise Exception("No suitable format found")

                    # Cache paths
                    url_hash = hashlib.md5(url.encode()).hexdigest()
                    video_path = self.video_cache.cache_dir / f"{url_hash}.mp4"
                    audio_path = self.video_cache.cache_dir / f"{url_hash}.mp3"

                    # Download video
                    video_opts = {
                        'format': best_format['format_id'],
                        'outtmpl': str(video_path),
                        'quiet': True,
                    }
                    with yt_dlp.YoutubeDL(video_opts) as ydl:
                        ydl.download([url])

                    # Download audio
                    audio_opts = {
                        'format': 'bestaudio',
                        'outtmpl': str(audio_path)[:-4],  # Remove .mp3 extension
                        'quiet': True,
                        'postprocessors': [{
                            'key': 'FFmpegExtractAudio',
                            'preferredcodec': 'mp3',
                        }]
                    }
                    with yt_dlp.YoutubeDL(audio_opts) as ydl:
                        ydl.download([url])

                    # Cache the video
                    self.video_cache.cache_video(url, str(video_path), str(audio_path))
                    
                    # Start playback
                    self.start_cached_video(str(video_path), str(audio_path))

            except Exception as e:
                self.status_label.setText("ERROR: Check URL")
                print(f"Detailed error: {str(e)}")

        except Exception as e:
            self.status_label.setText("ERROR: Invalid URL")
            print(f"Detailed error: {str(e)}")
        finally:
            self.play_button.setEnabled(True)

    def start_cached_video(self, video_path, audio_path):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait()

        try:
            # Stop any existing audio
            mixer.music.stop()
            mixer.music.unload()
            
            # Start audio first (only for YouTube videos)
            if video_path != audio_path:  # Not a local file
                mixer.music.load(audio_path)
                mixer.music.play()
            
            # Then start video
            self.video_thread = VideoThread(video_path, self.video_window)
            self.video_thread.frame_ready.connect(self.process_frame)
            self.video_thread.error.connect(lambda msg: self.status_label.setText(f"ERROR: {msg}"))
            self.video_thread.start()

            self.status_label.setText("PLAYING")
            
        except Exception as e:
            self.status_label.setText("ERROR: Failed to start playback")
            print(f"Playback error: {str(e)}")

    def closeEvent(self, event):
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread.wait()
        # Make sure to stop audio before closing
        mixer.music.stop()
        mixer.quit()
        
        # Clean up temporary files
        try:
            os.remove('temp_audio.mp3')
        except:
            pass
        
        self.video_window.close()
        event.accept()

    def process_frame(self, frame):
        if frame is None:
            return
            
        try:
            # Apply CRT effects
            frame = self.add_crt_effects(frame)
            
            # Convert to RGB for Qt
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create QImage
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, 
                           QImage.Format.Format_RGB888)
            
            # Scale to fill window while maintaining aspect ratio
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(
                self.video_window.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            # Update the video label
            self.video_window.video_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            print(f"Frame processing error: {str(e)}")

    def add_crt_effects(self, frame):
        # Convert to float32
        frame = frame.astype(np.float32) / 255.0
        
        # Scanlines with intensity control
        scanline_intensity = self.scanline_slider.value() / 100.0
        if scanline_intensity > 0:
            scanlines = np.ones_like(frame)
            scanlines[::2] *= max(0.1, 1.0 - scanline_intensity)  # Adjustable darkness
            scanlines[1::2] *= min(1.2, 1.0 + scanline_intensity * 0.2)  # Adjustable brightness
            frame = frame * scanlines
        
        # RGB Shift with intensity control
        rgb_shift = self.rgb_shift_slider.value() / 100.0
        if rgb_shift > 0:
            shift_amount = int(rgb_shift * 5)  # Up to 5 pixels shift
            frame_r = np.roll(frame[:,:,0], shift_amount, axis=1)
            frame_g = frame[:,:,1]
            frame_b = np.roll(frame[:,:,2], -shift_amount, axis=1)
            frame = np.dstack((frame_r, frame_g, frame_b))
        
        # Bloom effect
        bloom_intensity = self.bloom_slider.value() / 100.0
        if bloom_intensity > 0:
            bloom = cv2.GaussianBlur(frame, (0, 0), bloom_intensity * 3)
            frame = cv2.addWeighted(frame, 1.0, bloom, bloom_intensity, 0)
        
        # Noise
        noise_intensity = self.noise_slider.value() / 100.0
        if noise_intensity > 0:
            noise = np.random.normal(0, noise_intensity * 0.1, frame.shape)
            frame = np.clip(frame + noise, 0, 1)
        
        # Vignette
        vignette_intensity = self.vignette_slider.value() / 100.0
        if vignette_intensity > 0:
            rows, cols = frame.shape[:2]
            kernel_x = cv2.getGaussianKernel(cols, cols/2)
            kernel_y = cv2.getGaussianKernel(rows, rows/2)
            kernel = kernel_y * kernel_x.T
            mask = kernel / kernel.max()
            vignette = 1 - (vignette_intensity * (1 - mask))
            frame *= np.dstack([vignette] * 3)
        
        # Curvature (existing code)
        curvature = self.curvature_slider.value() / 100.0
        if curvature > 0:
            rows, cols = frame.shape[:2]
            cache_key = f"{rows}_{cols}_{curvature}"
            if not hasattr(self, 'curvature_cache'):
                self.curvature_cache = {}
                
            if cache_key not in self.curvature_cache:
                map_x, map_y = np.meshgrid(np.linspace(0, cols-1, cols),
                                         np.linspace(0, rows-1, rows))
                
                center_x, center_y = cols/2, rows/2
                dx = map_x - center_x
                dy = map_y - center_y
                d = np.sqrt(dx*dx + dy*dy)
                d2 = d*d
                
                # More aggressive edge darkening
                edge_darkness = np.clip(1 - (d2 / (rows*cols)) * 0.7, 0.6, 1.0)
                
                # More pronounced curvature
                f = 1 + d2 * curvature * 0.000005
                
                self.curvature_cache[cache_key] = (
                    (dx*f + center_x).astype(np.float32),
                    (dy*f + center_y).astype(np.float32),
                    edge_darkness.astype(np.float32)
                )
            
            map_x, map_y, edge_darkness = self.curvature_cache[cache_key]
            frame = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
            frame *= np.dstack([edge_darkness] * 3)
        
        # Final contrast adjustment
        frame = np.clip((frame - 0.5) * 1.2 + 0.5, 0, 1)
        
        return (frame * 255).astype(np.uint8)

    def open_local_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video File",
            "",
            "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*.*)"
        )
        
        if file_path:
            self.url_input.setText(file_path)
            self.start_video()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = ControlWindow()
    player.show()
    sys.exit(app.exec())